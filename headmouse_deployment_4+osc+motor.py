# This code was created through help from the Google Developers forum, specifically using Mediapipe (https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker), but also using Google's Gemini
# to create this. Specifically, this code is a combination of multiple things together; combining MediaPipe to extract facial expression data and head movements that are used to move a cursor. It combines Serial
# communication to communicate with an ESP32C6 and a NEMA17 stepper motor, OSC to communicate data to TouchDesigner & Ableton and uses NDIlib to use share video screen with Python & TouchDesigner. In doing so,
# it creates an interactive installation that is created for my Industrial Design thesis; where I looked how to situate oddness in public, normative contexts.

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import mouse
import keyboard
import numpy as np
import time
from pythonosc import udp_client, dispatcher, osc_server
import NDIlib as ndi
import threading
import serial
import pygetwindow as gw
import os
import pyautogui

# Configurations of MediaPipe
BaseOptions = mp_tasks.BaseOptions
FaceLandmarker = mp_vision.FaceLandmarker
FaceLandmarkerOptions = mp_vision.FaceLandmarkerOptions
VisionRunningMode = mp_vision.RunningMode

# PATH TO MEDIAPIPE
MODEL_PATH = //

# INITALIZATION TO MEDIAPIPE FACE LANDMARKER DETECTION
options_landmarker = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_faces=1,
    output_face_blendshapes=True
)

# SCREEN SIZE, CENTER POINT AND PREVIOUS MOUSE POINTS FOR CALCULATIONS
screen_w, screen_h = 1920, 1080
center_x, center_y = screen_w // 2, screen_h // 2
prev_move_x, prev_move_y = center_x, center_y

# SMOOTHING FOR MOUSE NAVIGATION AND MOVEMENTS
SCALE = 50
SMOOTHING_NORMAL = 0.10
SMOOTHING_CLICKING = 0.005
current_smoothing = 0

right_timestamps = []
FRAME_WINDOW_MS = 200
MIN_FRAMES = 5

prev_face_detected = False

# FROM TOUCHDESIGNER
td_state = 0
prev_td_state = 0
td_triggered = False

#FOR CURTAIN
ser = serial.Serial('COM5', 115200)
curtain_status = True  # False for Closed, True for Open
face_first_seen_time = 0
curtain_released = False
last_face_time = int(time.time() * 1000)
INACTIVITY_THRESHOLD = 5000

# MOUSE CONFIGURATIONS FAILSAFE
cursor_active = True
FAILSAFE_KEY = 'f12'
keyboard.add_hotkey(FAILSAFE_KEY, lambda: print("Toggled cursor"))

# MOUSE CLICK STATES: to calculate the time between mouse clicks.
is_left_clicked = False
left_press_time = left_release_time = left_release_check_time = 0
is_right_clicked = False
right_press_time = right_release_time = right_release_check_time = 0
# --------------------------------------------------------------------------- #

def handle_from_td(address, *args):
    global td_state, prev_td_state, td_triggered
    td_state = args[0]
    if prev_td_state == 0 and td_state == 1:
        td_triggered = True
        prev_td_state = 1
        pyautogui.hotkey('ctrl', 'win', 'right')

    if prev_td_state == 1 and td_state == 0:
        prev_td_state = 0
    

# FIRST FUNCTION FOR MOUSE NAVIGATION: uses machine learning model to detect face (face landmarks), place it on
# a face mesh and tracks 6 points to be able to track head pose to navigate the mouse.
def process_pose_from_landmarker(image, results):
    global prev_move_x, prev_move_y, current_smoothing
    img_h, img_w, _ = image.shape
    face_3d, face_2d = [], []

    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            for idx, lm in enumerate(face_landmarks):
                if idx in [33, 263, 1, 61, 291, 199]:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

        if len(face_2d) < 6:
            return

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = img_w
        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                               [0, focal_length, img_w / 2],
                               [0, 0, 1]])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success_pnp, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        x_angle = angles[0] * 360
        y_angle = angles[1] * 360

        move_x = int(center_x + (y_angle * SCALE))
        move_y = int(center_y + (-x_angle * SCALE))
        move_x = max(0, min(screen_w-1, move_x))
        move_y = max(0, min(screen_h-1, move_y))

        new_move_x = int(current_smoothing * move_x + (1 - current_smoothing) * prev_move_x)
        new_move_y = int(current_smoothing * move_y + (1 - current_smoothing) * prev_move_y)


        client = udp_client.SimpleUDPClient("127.0.0.1", 10001)
        client.send_message("/headpose", [new_move_x, new_move_y])

        prev_move_x = new_move_x
        prev_move_y = new_move_y

        if cursor_active:
            mouse.move(new_move_x, new_move_y, absolute=True)

# SECOND FUNCTION FOR MOUSE CLICKS: uses a different function (blendshapes) that can detect specific facial
# gestures, in this case: left and right winking and tensing the sides of the mouth. 
def process_blendshapes_sync(result, current_time):
    global is_left_clicked, left_press_time, left_release_time, left_release_check_time
    global is_right_clicked, right_press_time, right_release_time, right_release_check_time
    global current_smoothing, right_timestamps

    # Assign the values of left blink and right blink to respective variables.
    leftBlink_score = rightBlink_score = 0.0
    leftMouth_score = rightMouth_score = 0.0
    if result.face_blendshapes:
        blendshapes = result.face_blendshapes[0]
        for category in blendshapes:
            if category.category_name == "eyeBlinkRight":
                leftBlink_score = category.score
            if category.category_name == "eyeBlinkLeft":
                rightBlink_score = category.score
            if category.category_name == "mouthRight":
                rightMouth_score = category.score
            if category.category_name == "mouthLeft":
                leftMouth_score = category.score
    
    client = udp_client.SimpleUDPClient("127.0.0.1", 10001)
    client.send_message("/RightSide", [rightBlink_score, leftMouth_score])
    client.send_message("/LeftSide", [leftBlink_score, rightMouth_score])

    # RIGHT EYE
    if rightBlink_score > 0.35:
        right_timestamps.append(current_time)

    # Remove old timestamps
    right_timestamps = [t for t in right_timestamps if current_time - t <= FRAME_WINDOW_MS]


    # Assign a boolean to new variables if they have crossed a specific threshold.
    EXPRESSION_LEFT = (leftBlink_score > 0.35 and cursor_active)
    EXPRESSION_RIGHT = (len(right_timestamps) >= MIN_FRAMES and cursor_active)

    # LEFT CLICK LOGIC: leaves a little time between clicks to prevent instant double click.
    if EXPRESSION_LEFT:
        current_smoothing = SMOOTHING_CLICKING  #To slow down the mouse while clicking.
        if left_release_check_time > 0: left_release_check_time = 0
        if not is_left_clicked:
            mouse.press(button="left")
            is_left_clicked = True
            left_press_time = current_time
    elif is_left_clicked and not EXPRESSION_LEFT:
        current_smoothing = SMOOTHING_NORMAL   #Back to normal mouse speed.
        if left_release_check_time == 0:
            left_release_check_time = current_time
        elif current_time - left_release_check_time >= 100:
            mouse.release(button="left")
            left_release_time = current_time
            is_left_clicked = False
            left_release_check_time = 0

    # RIGHT CLICK
    if EXPRESSION_RIGHT:
        current_smoothing = SMOOTHING_CLICKING
        if right_release_check_time > 0: right_release_check_time = 0
        if not is_right_clicked:
            mouse.press(button="right")
            is_right_clicked = True
            right_press_time = current_time
    elif is_right_clicked and not EXPRESSION_RIGHT:
        current_smoothing = SMOOTHING_NORMAL
        if right_release_check_time == 0:
            right_release_check_time = current_time
        elif current_time - right_release_check_time >= 100:
            mouse.release(button="right")
            right_release_time = current_time
            is_right_clicked = False
            right_release_check_time = 0

# --- MAIN PROGRAM --- #
with FaceLandmarker.create_from_options(options_landmarker) as landmarker:
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Haal de resolutie op van je webcam
    width  = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ndi.initialize()
    send_settings = ndi.SendCreate()
    send_settings.ndi_name = "HeadmouseFeed"
    sender = ndi.send_create(send_settings)

    disp = dispatcher.Dispatcher()
    disp.map("/from_TD", handle_from_td)
    server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 10003), disp)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    print("OSC server luistert op poort 10003")

    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break

        flipped_frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = landmarker.detect(mp_image)

        current_time = int(time.time() * 1000)
        process_pose_from_landmarker(flipped_frame, results)
        process_blendshapes_sync(results, current_time)
        face_detected = bool(results.face_landmarks)

        client = udp_client.SimpleUDPClient("127.0.0.1", 10001)

        if face_detected and not prev_face_detected:
            print("FACE JUST DETECTED")
            client.send_message("/face_detected", 1)

        if not face_detected and prev_face_detected:
            print("FACE LOST")
            client.send_message("/face_detected", 0)

        prev_face_detected = face_detected

        # ← Stuur frame naar virtuele webcam
        frame = ndi.VideoFrameV2()
        frame.data = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2BGRA)
        frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRA
        ndi.send_send_video_v2(sender, frame)


        if face_detected:
            last_face_time = current_time

            if face_first_seen_time == 0:
                face_first_seen_time = current_time
                ser.write(b'r\n')
                #print("gordijn reset")

            if curtain_status == True and not curtain_released:
                ser.write(b'd\n')
                #print("gordijn dicht")
                curtain_status = False

            if curtain_status == False and not curtain_released:
                if td_triggered:
                    ser.write(b'o\n')
                    #print("gordijn open - vrijgegeven door TD")
                    curtain_status = True
                    curtain_released = True
                    td_triggered = False  # ← reset de vlag
        else:
            time_without_face = current_time - last_face_time

            if time_without_face >= INACTIVITY_THRESHOLD:
                face_first_seen_time = 0

                if curtain_status == False:
                    ser.write(b'o\n')
                    #print("gordijn open - persoon weg")
                    curtain_status = True
                    curtain_released = False
                    pyautogui.hotkey('ctrl', 'win', 'left')

                elif curtain_released:
                    curtain_released = False
                    #print("reset voor de volgende persoon")
                    pyautogui.hotkey('ctrl', 'win', 'left')

        cv2.imshow('Headmouse', flipped_frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ndi.send_destroy(sender)
    ndi.destroy()

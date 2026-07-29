import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import mouse
import keyboard
import numpy as np
import time
import os

# --- CONFIGURATIE --- #
BaseOptions = mp_tasks.BaseOptions
FaceLandmarker = mp_vision.FaceLandmarker
FaceLandmarkerOptions = mp_vision.FaceLandmarkerOptions
VisionRunningMode = mp_vision.RunningMode

MODEL_PATH = 'C:/Users/Roy/OneDrive/Master/M2.1/Project/Exploration/Face_track_map/face_landmarker.task'

cursor_active = True
FAILSAFE_KEY = 'f12'
keyboard.add_hotkey(FAILSAFE_KEY, lambda: print("Toggled cursor"))

# Video paths
script_dir = os.path.dirname(os.path.abspath(__file__))
idle_video_path = os.path.join(script_dir, "Blinking_Eye.mp4")
welcome_video_path = os.path.join(script_dir, "Just_Tutorial1.mp4")

print("Idle video pad:", idle_video_path)
print("Welcome video pad:", welcome_video_path)

# Video state
video_cap = None
screensaver_active = False
welcome_active = False

INACTIVITY_THRESHOLD = 2000  # ms
last_face_time = int(time.time() * 1000)

# Screen and mouse
screen_w, screen_h = 1920, 1080
center_x, center_y = screen_w // 2, screen_h // 2
prev_move_x, prev_move_y = center_x, center_y
SCALE = 50
SMOOTHING_NORMAL = 0.04
SMOOTHING_CLICKING = 0.005
current_smoothing = 0

# Click state
is_left_clicked = False
left_press_time = left_release_time = left_release_check_time = 0
is_right_clicked = False
right_press_time = right_release_time = right_release_check_time = 0

# --- INIT FACE LANDMARKER --- #
options_landmarker = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_faces=1,
    output_face_blendshapes=True
)

# --- FUNCTIES --- #
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

        prev_move_x = new_move_x
        prev_move_y = new_move_y

        if cursor_active and not (is_left_clicked or is_right_clicked):
            mouse.move(new_move_x, new_move_y, absolute=True)

def process_blendshapes_sync(result, current_time):
    global is_left_clicked, left_press_time, left_release_time, left_release_check_time
    global is_right_clicked, right_press_time, right_release_time, right_release_check_time
    global current_smoothing

    leftBlink_score = rightBlink_score = 0.0

    if result.face_blendshapes:
        blendshapes = result.face_blendshapes[0]
        for category in blendshapes:
            if category.category_name == "eyeBlinkRight":
                leftBlink_score = category.score
            if category.category_name == "eyeBlinkLeft":
                rightBlink_score = category.score

    EXPRESSION_LEFT = (leftBlink_score > 0.35 and cursor_active)
    EXPRESSION_RIGHT = (rightBlink_score > 0.35 and cursor_active)

    # LEFT CLICK
    if EXPRESSION_LEFT:
        current_smoothing = SMOOTHING_CLICKING
        if left_release_check_time > 0: left_release_check_time = 0
        if not is_left_clicked:
            mouse.press(button="left")
            is_left_clicked = True
            left_press_time = current_time
    elif is_left_clicked and not EXPRESSION_LEFT:
        current_smoothing = SMOOTHING_NORMAL
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
    video_capture = cv2.VideoCapture(0)  # webcam

    # --- START IDLE VIDEO BIJ OPSTART --- #
    screensaver_active = True
    welcome_active = False
    video_cap = cv2.VideoCapture(idle_video_path)
    if not video_cap.isOpened():
        print(f"⚠️ Kan video niet openen: {idle_video_path}")

    # Toon eerste frame direct
    if video_cap:
        ret, vframe = video_cap.read()
        if ret:
            cv2.namedWindow("Idle Video", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Idle Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Idle Video", vframe)

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

        # --- PRESENCE BASED VIDEO LOGIC --- #
        if results.face_landmarks:
            last_face_time = current_time
            # Stop idle video
            if screensaver_active and video_cap:
                video_cap.release()
                try:
                    cv2.destroyWindow("Idle Video")
                except cv2.error:
                    pass
                video_cap = None
                screensaver_active = False
            # Start welcome video
            if not welcome_active:
                welcome_active = True
                video_cap = cv2.VideoCapture(welcome_video_path)
                if not video_cap.isOpened():
                    print(f"⚠️ Kan video niet openen: {welcome_video_path}")
        else:
            if current_time - last_face_time > INACTIVITY_THRESHOLD:
                if not screensaver_active:
                    screensaver_active = True
                    welcome_active = False
                    video_cap = cv2.VideoCapture(idle_video_path)

        # --- RENDER VIDEO --- #
        # --- RENDER VIDEO --- #
        if video_cap:
            ret, vframe = video_cap.read()
            if not ret:
                # Video afgelopen
                if welcome_active:
                    try:
                        cv2.destroyWindow("Welcome Video")
                    except cv2.error:
                        pass
                    video_cap = None
                    welcome_active = False
            # Hier laten we scherm verder open voor interactie (geen blocking)
            elif screensaver_active:
                # Idle video loop
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, vframe = video_cap.read()
        if ret:
            wname = "Welcome Video" if welcome_active else "Idle Video"
            cv2.namedWindow(wname, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(wname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(wname, vframe)

        cv2.imshow('Headmouse', flipped_frame)
        if cv2.waitKey(30) & 0xFF == 27:  # ESC
            break

    video_capture.release()
    if video_cap:
        video_cap.release()
    cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import os
import math
import time

# -------------------------------
# Initialize MediaPipe
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)

# -------------------------------
# Load Glasses Assets
# -------------------------------
glasses_folder = "glass_assets"
glasses_list = [
    cv2.imread(os.path.join(glasses_folder, f), cv2.IMREAD_UNCHANGED)
    for f in os.listdir(glasses_folder)
    if f.endswith(".png")
]
if not glasses_list:
    raise Exception("No glasses images found in 'glass_assets' folder!")

current_glasses = 0
prev_hand_open = False
last_switch_time = 0
cooldown = 0.6  # seconds before allowing another switch

# -------------------------------
# Helper Functions
# -------------------------------
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rotated

def overlay_image(background, overlay, x, y, w, h):
    overlay_resized = cv2.resize(overlay, (w, h))
    b, g, r, a = cv2.split(overlay_resized)
    mask = a / 255.0
    for c in range(3):
        background[y:y+h, x:x+w, c] = (1 - mask) * background[y:y+h, x:x+w, c] + mask * overlay_resized[:, :, c]
    return background

def is_hand_open(landmarks):
    """Detect open palm using relative finger tip positions."""
    tips = [4, 8, 12, 16, 20]
    open_count = 0
    for tip in tips[1:]:  # skip thumb
        if landmarks[tip].y < landmarks[tip - 2].y:
            open_count += 1
    return open_count >= 3

# -------------------------------
# Setup Camera
# -------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
cv2.namedWindow("Glasses Filter", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Glasses Filter", 960, 540)

print("🖐️ Show open palm to switch glasses. Press 'q' to quit.")
frame_counter = 0

# -------------------------------
# Main Loop
# -------------------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process face and hands
    face_result = face_mesh.process(frame_rgb)
    frame_counter += 1
    hand_result = None
    if frame_counter % 2 == 0:
        hand_result = hands.process(frame_rgb)

    # ---------------- Face Overlay ----------------
    if face_result.multi_face_landmarks:
        for face_landmarks in face_result.multi_face_landmarks:
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            x1, y1 = int(left_eye.x * w), int(left_eye.y * h)
            x2, y2 = int(right_eye.x * w), int(right_eye.y * h)

            angle_rad = math.atan2(y2 - y1, x2 - x1)
            angle_deg = -math.degrees(angle_rad)

            glasses_width = int(1.6 * abs(x2 - x1))
            glasses_height = int(glasses_width * 0.5)
            x = int((x1 + x2) / 2 - glasses_width / 2)
            y = int((y1 + y2) / 2 - glasses_height / 2)

            x, y = max(0, x), max(0, y)
            if x + glasses_width > w:
                glasses_width = w - x
            if y + glasses_height > h:
                glasses_height = h - y

            rotated_glasses = rotate_image(glasses_list[current_glasses], angle_deg)
            frame = overlay_image(frame, rotated_glasses, x, y, glasses_width, glasses_height)

    # ---------------- Hand Gesture Control ----------------
    hand_open = False
    if hand_result and hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            if is_hand_open(hand_landmarks.landmark):
                hand_open = True

    # Only trigger once when gesture changes from closed → open
    current_time = time.time()
    if hand_open and not prev_hand_open and (current_time - last_switch_time > cooldown):
        current_glasses = (current_glasses + 1) % len(glasses_list)
        last_switch_time = current_time
        print(f"🔁 Switched to glasses {current_glasses + 1}")

    prev_hand_open = hand_open

    # ---------------- Display ----------------
    cv2.imshow("Glasses Filter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

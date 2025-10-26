import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# LED state
led_state = False
led_position = (100, 200)  # Moved below the status text (which is at y=50)
led_radius = 40

# Track previous state for each hand
hands_prev_state = {}  # key: hand index, value: True if closed, False if open
hold_time_required = 0.2  # seconds
hands_closed_start = {}  # time when hand closed started

def is_closed_palm(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]
    closed = 0
    for tip, pip in zip(tips_ids, pip_ids):
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
            closed += 1
    return closed >= 4

def draw_incandescent_led(frame, position, radius, state):
    x, y = position
    overlay = frame.copy()
    
    if state:
        # Base flicker effect with warm color
        flicker_intensity = np.random.randint(200, 255)
        base_color = (20, 100 + flicker_intensity//3, flicker_intensity)  # More yellow-orange
        
        # Draw bulb glass (elongated ellipse for side view)
        bulb_center = (x, y - radius//2)  # Moved up
        axes = (int(radius//1.3), radius)  # Width, height of ellipse
        cv2.ellipse(overlay, bulb_center, axes, 0, 0, 360, (255, 255, 255), 2)
        
        # Draw metal base (socket) at bottom
        base_points = np.array([
            [x - radius//2, y + radius],      # Left bottom
            [x - radius//3, y + radius//2],    # Left middle
            [x - radius//3, y],               # Left top
            [x + radius//3, y],               # Right top
            [x + radius//3, y + radius//2],    # Right middle
            [x + radius//2, y + radius],      # Right bottom
        ], np.int32)
        cv2.fillPoly(overlay, [base_points], (120, 120, 120))  # Metal color
        
        # Draw filament (more complex for side view)
        filament_points = np.array([
            [x - radius//4, y - radius//2],  # Start
            [x - radius//8, y - radius*3//4],  # Up
            [x + radius//8, y - radius//3],  # Down
            [x + radius//4, y - radius//2]  # End
        ], np.int32)
        cv2.polylines(overlay, [filament_points], False, base_color, 2)
        
        # Inner glow (filament glow)
        inner_overlay = np.zeros_like(frame, dtype=np.uint8)
        cv2.ellipse(inner_overlay, bulb_center, 
                   (int(axes[0]-2), int(axes[1]-2)), 0, 0, 360, base_color, -1)
        cv2.addWeighted(inner_overlay, 0.6, overlay, 1, 0, overlay)
        
        # Multiple layer glow effect with different colors
        glow_colors = [
            ((20, 100 + flicker_intensity//3, flicker_intensity), 0.4),  # Inner warm
            ((10, 50 + flicker_intensity//4, flicker_intensity), 0.3),   # Middle
            ((0, 30 + flicker_intensity//5, flicker_intensity//2), 0.2)  # Outer
        ]
        
        for i, (color, alpha) in enumerate(glow_colors):
            glow_axes = (int(float(axes[0]) * (1 + (i+1) * 0.5)), 
                        int(float(axes[1]) * (1 + (i+1) * 0.5)))
            temp = np.zeros_like(frame, dtype=np.uint8)
            cv2.ellipse(temp, bulb_center, glow_axes, 0, 0, 360, color, -1)
            temp = cv2.GaussianBlur(temp, (15, 15), 10)
            cv2.addWeighted(temp, alpha, overlay, 1, 0, overlay)
            
        # Add highlight reflection (elongated for side view)
        highlight = np.zeros_like(frame, dtype=np.uint8)
        highlight_center = (x - radius//4, y - radius*2//3)
        cv2.ellipse(highlight, highlight_center, 
                   (radius//6, radius//3), -45, 0, 360, (255, 255, 255), -1)
        highlight = cv2.GaussianBlur(highlight, (9, 9), 5)
        cv2.addWeighted(highlight, 0.3, overlay, 1, 0, overlay)
    else:
        # Unlit bulb appearance
        # Draw bulb glass (elongated ellipse)
        bulb_center = (x, y - radius//2)
        axes = (int(radius//1.3), radius)
        cv2.ellipse(overlay, bulb_center, axes, 0, 0, 360, (180, 180, 180), 2)
        
        # Draw metal base (socket) at bottom
        base_points = np.array([
            [x - radius//2, y + radius],      # Left bottom
            [x - radius//3, y + radius//2],    # Left middle
            [x - radius//3, y],               # Left top
            [x + radius//3, y],               # Right top
            [x + radius//3, y + radius//2],    # Right middle
            [x + radius//2, y + radius],      # Right bottom
        ], np.int32)
        cv2.fillPoly(overlay, [base_points], (100, 100, 100))
        
        # Draw dark filament
        filament_points = np.array([
            [x - radius//4, y - radius//2],
            [x - radius//8, y - radius*3//4],
            [x + radius//8, y - radius//3],
            [x + radius//4, y - radius//2]
        ], np.int32)
        cv2.polylines(overlay, [filament_points], False, (100, 100, 100), 2)
        
        # Add slight inner shadow
        inner_shadow = np.zeros_like(frame, dtype=np.uint8)
        cv2.ellipse(inner_shadow, bulb_center, 
                   (int(axes[0]-2), int(axes[1]-2)), 0, 0, 360, (30, 30, 30), -1)
        cv2.addWeighted(inner_shadow, 0.3, overlay, 1, 0, overlay)
    
    frame[:] = overlay

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        wrist_closed = False
        current_time = time.time()

        current_hands = {}
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                closed = is_closed_palm(hand_landmarks)
                current_hands[idx] = closed

                # Detect transition: open → closed
                prev_state = hands_prev_state.get(idx, False)
                if closed and not prev_state:
                    # Start counting hold time
                    hands_closed_start[idx] = current_time
                elif not closed:
                    hands_closed_start.pop(idx, None)

                # Toggle LED if hand was closed and hold time exceeded
                if closed and idx in hands_closed_start:
                    if current_time - hands_closed_start[idx] >= hold_time_required:
                        led_state = not led_state
                        print(f"Hand {idx} Closed → LED TOGGLED")
                        hands_closed_start.pop(idx)
                wrist_closed = wrist_closed or closed

        hands_prev_state = current_hands.copy()

        # Draw LED
        draw_incandescent_led(frame, led_position, led_radius, led_state)

        # Status text
        status_text = "Wrist Closed" if wrist_closed else "Wrist Open"
        cv2.putText(frame, f"{status_text}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        # print(status_text)

        cv2.imshow("Incandescent LED Hand Toggle (State Change)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

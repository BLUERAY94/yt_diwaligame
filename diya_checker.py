import cv2
import mediapipe as mp
import numpy as np
import random, time, os, pygame

pygame.mixer.init()

# Configuration
CAM_SOURCE = 0  # Default camera
GAME_TIME = 50 #Game duration in seconds
START_SPAWN_INTERVAL = 1  # Initial spawn interval in seconds
MIN_SPAWN_INTERVAL = 0.3  # Minimum spawn interval in seconds
START_GRAVITY = 2  # Initial gravity
MAX_GRAVITY = 8  # Maximum gravity
DIYA_SCORE = 10  # Points per diya
SWEET_SCORE = 25  # Points per sweet
FIRECRACKER_PENALTY = 15  # Penalty for firecracker
OBJECT_SIZE = 80  # Size of objects
CATCH_RADIUS = 100  # Catch radius in pixels


ENABLE_SOUND = True
try:
    pygame.mixer.init()
    diya_sound    = pygame.mixer.Sound(os.path.join('sounds', 'lamp.wav'))
    sweet_sound   = pygame.mixer.Sound(os.path.join('sounds', 'sweet.wav'))
    cracker_sound = pygame.mixer.Sound(os.path.join('sounds', 'cracker.wav'))
except Exception as e:
    print(f"Sound initialization failed: {e}")
    ENABLE_SOUND = False

def play_sound(sound):
    if ENABLE_SOUND:
        try:
            sound.play()
        except Exception as e:
            print(f"Error playing sound: {e}")

def load_png_with_alpha(path, size):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    if img.shape[2] == 3:  # Add alpha only if missing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    if size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img



DIYA_IMG        = load_png_with_alpha(os.path.join('images', 'lamp.png'), (OBJECT_SIZE, OBJECT_SIZE))
FIRECRACKER_IMG = load_png_with_alpha(os.path.join('images', 'firecracker.png'), (OBJECT_SIZE, OBJECT_SIZE))
SWEET_IMG       = load_png_with_alpha(os.path.join('images', 'sweet.png'), (OBJECT_SIZE, OBJECT_SIZE))
BURST_IMG       = load_png_with_alpha(os.path.join('images', 'burst.png'), (OBJECT_SIZE, OBJECT_SIZE))

mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw  = mp.solutions.drawing_utils

class FallingObject:
    def __init__(self, kind, x, y, gravity=START_GRAVITY):
        self.kind = kind  # 'diya', 'sweet', 'firecracker'
        if kind == 'diya':
            self.image = DIYA_IMG
        elif kind == 'sweet':
            self.image = SWEET_IMG
        elif kind == 'firecracker':
            self.image = FIRECRACKER_IMG

        self.x = x
        self.y = y
        self.size = self.image.shape[0]
        self.vy = 0
        self.ay = gravity
        self.vx = random.uniform(-1, 1)
        self.max_vy = 30

    def move(self):
        self.vy += self.ay
        self.vy = min(self.vy, self.max_vy)
        self.y += self.vy
        self.x += self.vx

    def draw(self, frame):
        pos = (int(self.x), int(self.y))
        overlay_image(frame, self.image, pos)

    def is_off_screen(self, height):
        return self.y > height

def overlay_image(bg, fg, pos):
    x, y = pos
    h, w = fg.shape[:2]

    # Check bounds
    if x < 0 or y < 0 or x + w > bg.shape[1] or y + h > bg.shape[0]:
        return

    # Separate RGBA channels
    if fg.shape[2] == 4:
        alpha = fg[:, :, 3] / 255.0
        alpha = alpha[..., np.newaxis]  # Expand to (h, w, 1)
        fg_rgb = fg[:, :, :3]

        # Blend
        bg[y:y + h, x:x + w] = (alpha * fg_rgb +
                                (1 - alpha) * bg[y:y + h, x:x + w])
    else:
        bg[y:y + h, x:x + w] = fg


def is_caught(obj, hand_circle):
    ox = int(obj.x + obj.size / 2)
    oy = int(obj.y + obj.size / 2)
    center_x, center_y, radius = hand_circle
    obj_radius = obj.size / 2
    dist = np.sqrt((center_x - ox) ** 2 + (center_y - oy) ** 2)
    return dist < (radius + obj_radius)


def is_closed_palm(hand_landmarks, frame_shape):
    h, w = frame_shape[:2]
    palm_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * w)
    palm_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * h)
    tips = [4, 8, 12, 16, 20]  # Thumb tip, Index tip, Middle tip, Ring tip, Pinky tip
    close_count = 0

    for tip in tips:
        tip_x = int(hand_landmarks.landmark[tip].x * w)
        tip_y = int(hand_landmarks.landmark[tip].y * h)
        distance = np.sqrt((tip_x - palm_x) ** 2 + (tip_y - palm_y) ** 2)
        if distance < 50:  # Threshold distance to consider finger closed
            close_count += 1
    return close_count >= 4  # All fingers closed


def get_hand_circle(hand_landmarks, frame_shape):
    h, w, _ = frame_shape
    xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
    ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
    center_x = int(np.mean(xs))
    center_y = int(np.mean(ys))

    radius = int(max(np.std(xs), np.std(ys))) + 30  # Add margin
    radius = min(radius, CATCH_RADIUS)  # Cap radius to CATCH_RADIUS
    return center_x, center_y, radius

def draw_hand_circle(frame, hand_landmarks):
    center_x, center_y, radius = get_hand_circle(hand_landmarks, frame.shape)
    cv2.circle(frame, (center_x, center_y), radius, (255, 0, 255), 2)

def run_game_session(cap, hands):
    try:
        pygame.mixer.music.load(os.path.join('sounds', 'background.mp3'))
        pygame.mixer.music.set_volume(0.5)
        pygame.mixer.music.play(-1)
    
    except Exception as e:
        print(f"Background music error: {e}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    window_name = "Happy Diwali!"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    score = 0
    objects = []
    last_spawn_time = time.time()
    start_time = time.time()
    burst_effects = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_circles = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                draw_hand_circle(frame, hand_landmarks)
                hand_circles.append(get_hand_circle(hand_landmarks, frame.shape))

        progress = min(score / 200.0, 1.0)  # 0 to 1 scaling
        spawn_interval = START_SPAWN_INTERVAL - (START_SPAWN_INTERVAL - MIN_SPAWN_INTERVAL) * progress
        gravity = START_GRAVITY + (MAX_GRAVITY - START_GRAVITY) * progress

        now = time.time()
        if now - last_spawn_time > spawn_interval:
            rand = random.random()
            if rand < 0.6:
                kind = 'diya'
            elif rand < 0.9:
                kind = 'firecracker'
            else:
                kind = 'sweet'
            
            x = random.randint(0, width - OBJECT_SIZE)
            obj = FallingObject(kind, x, 0, gravity)
            objects.append(obj)
            last_spawn_time = now

        for obj in objects[:]:
            obj.move()
            obj.draw(frame)
            caught = False

            for hand_circle in hand_circles:
                if is_caught(obj, hand_circle):
                    if obj.kind == 'diya':
                        score += DIYA_SCORE
                        play_sound(diya_sound)
                    elif obj.kind == 'sweet':
                        score += SWEET_SCORE
                        play_sound(sweet_sound)
                    elif obj.kind == 'firecracker':
                        score -= FIRECRACKER_PENALTY
                        play_sound(cracker_sound)
                        burst_effects.append((int(obj.x), int(obj.y), time.time()))
                    objects.remove(obj)
                    caught = True
                    break
            if not caught and obj.is_off_screen(height):
                objects.remove(obj)

        burst_effects = [b for b in burst_effects if time.time() - b[2] < 0.5]
        for bx, by, bt in burst_effects:
            overlay_image(frame, BURST_IMG, (bx, by))
        
        elapsed = int(now - start_time)
        remaining = max(0, GAME_TIME - elapsed)

        cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(frame, f"Time: {remaining}s", (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == 27 or remaining==0:  # ESC to quit
            break

    pygame.mixer.music.stop()
    print(f"Final Score: {score}")
    play_again = show_game_over_handtrackable(score, cap, hands, window_name)
    return play_again

def show_game_over_handtrackable(score, cap, hands, window_name):
    try:
        pygame.mixer.music.load(os.path.join('sounds', 'game_over.mp3'))
        pygame.mixer.music.play(-1)
    except Exception as e:
        print(f"Game over music error: {e}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    button_height = 70
    button_width  = 220
    gap = 40

    play_rect = (width//2 - button_width - gap//2, height-button_height-40, width//2-gap//2, height-40)
    exit_rect = (width//2 + gap//2, height-button_height-40, width//2 + button_width + gap//2, height-40)
    cooldown_start = time.time()
    cooldown_duration = 2.0  # seconds

    gesture_start_time = None
    gesture_target = None
    gesture_target_duration = 2.0  # seconds

    while True:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        x, y = 50, height // 3  # Left position and vertical base

        cv2.putText(img, "Game Over!", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, f"Your Score: {score}", (x, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 2, cv2.LINE_AA)

        play_color = (0, 200, 0)
        exit_color = (0, 0, 200)
        selection = None

        cv2.rectangle(img, (play_rect[0], play_rect[1]), (play_rect[2], play_rect[3]), play_color, -1)
        cv2.rectangle(img, (exit_rect[0], exit_rect[1]), (exit_rect[2], exit_rect[3]), exit_color, -1)
        cv2.putText(img, "Play Again", (play_rect[0]+30, play_rect[1]+45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(img, "Exit", (exit_rect[0]+70, exit_rect[1]+45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            small_h, small_w = 160, 120
            img[20:20+small_h, width-small_w-20:width-20] = cv2.resize(frame, (small_w, small_h))

        now = time.time()
        if now - cooldown_start < cooldown_duration:
            remaining_cooldown = int(cooldown_duration - (now - cooldown_start))
            cv2.putText(img, f"Please wait... {remaining_cooldown}s", (width//2 - 150, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow(window_name, img)
        
            if cv2.waitKey(30) & 0xFF == 27:
                cv2.destroyAllWindows()
                pygame.mixer.music.stop()
                return False
            
            gesture_start_time = None
            gesture_target = None
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None
        results = hands.process(rgb) if ret else None
        gesture_this_frame = None

        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
                ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
                
                for x, y in zip(xs, ys):
                    cv2.circle(img, (x, y), 5, (255, 0, 255), -1)
                    if play_rect[0] < x < play_rect[2] and play_rect[1] < y < play_rect[3]:
                        selection = 'play'
                    elif exit_rect[0] < x < exit_rect[2] and exit_rect[1] < y < exit_rect[3]:
                        selection = 'exit'
                
                if selection == 'play':
                    cv2.rectangle(img, (play_rect[0], play_rect[1]), (play_rect[2], play_rect[3]), (0,255,0), 4)
                elif selection == 'exit':
                    cv2.rectangle(img, (exit_rect[0], exit_rect[1]), (exit_rect[2], exit_rect[3]), (0,255,0), 4)

                if selection and is_closed_palm(hand_landmarks, img.shape):
                    gesture_this_frame = selection

        if gesture_this_frame:
            if gesture_target == gesture_this_frame:
                if gesture_start_time is not None and (now - gesture_start_time)>=gesture_target_duration:
                    if gesture_target == 'play':
                        cv2.putText(img, "Starting New Game!", (width//2 - 150, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        cv2.imshow(window_name, img)
                        cv2.waitKey(1000)
                        cv2.destroyAllWindows()
                        pygame.mixer.music.stop()
                        return True
                    elif gesture_target == 'exit':
                        cv2.putText(img, "Exiting...", (width//2 - 100, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        cv2.imshow(window_name, img)
                        cv2.waitKey(1000)
                        cv2.destroyAllWindows()
                        pygame.mixer.music.stop()
                        return False
                else:
                    elapsed = now - gesture_start_time if gesture_start_time else 0
                    bar_len = int(200 * min(elapsed / gesture_target_duration, 1))
                    bar_x = width//2 - 100
                    bar_y = height - button_height - 80
                    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_len, bar_y + 20), (0,255,0) if gesture_target == 'play' else (0,0,255), -1)
                    cv2.rectangle(img, (bar_x, bar_y), (bar_x + 200, bar_y + 20), (255,255,255), 2)

            else:
                gesture_target = gesture_this_frame
                gesture_start_time = now

        else:
            gesture_target = None
            gesture_start_time = None

        cv2.imshow(window_name, img)
        if cv2.waitKey(30) & 0xFF == 27:
            cv2.destroyAllWindows()
            pygame.mixer.music.stop()
            return False
    

if __name__ == "__main__":
    cap = cv2.VideoCapture(CAM_SOURCE)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    while True:
        play_again = run_game_session(cap, hands)
        if not play_again:
            break
    cap.release()
    cv2.destroyAllWindows()
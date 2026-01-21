import cv2
import mediapipe as mp
import os
import time

# -------------------- CONFIG --------------------

DATASET_PATH = "cnn_dataset"
LABEL = "Z"           # CHANGE THIS (A, B, C, ...)
NUM_IMAGES = 500      # images to collect
IMG_SIZE = 224        # CNN input size
SAVE_EVERY = 3        # save every N frames

# -------------------- SETUP --------------------

os.makedirs(os.path.join(DATASET_PATH, LABEL), exist_ok=True)

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

count = 0
frame_count = 0

print(f"Collecting data for label: {LABEL}")

# -------------------- MAIN LOOP --------------------

while count < NUM_IMAGES:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        h, w, _ = frame.shape

        x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
        y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

        x_min, x_max = max(min(x_coords) - 20, 0), min(max(x_coords) + 20, w)
        y_min, y_max = max(min(y_coords) - 20, 0), min(max(y_coords) + 20, h)

        hand_img = frame[y_min:y_max, x_min:x_max]

        if hand_img.size != 0:
            hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))

            frame_count += 1

            if frame_count % SAVE_EVERY == 0:
                img_path = os.path.join(
                    DATASET_PATH,
                    LABEL,
                    f"{count:04d}.jpg"
                )
                cv2.imwrite(img_path, hand_img)
                count += 1

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

    cv2.putText(frame, f"Label: {LABEL}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.putText(frame, f"Images: {count}/{NUM_IMAGES}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

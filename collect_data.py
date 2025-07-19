import cv2
import os
import time
import argparse
import numpy as np
import mediapipe as mp
from utils.mediapipe_utils import mediapipe_detection, extract_keypoints, draw_landmarks

# ----------------------------
# Handle command-line arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Collect sign language data.")
parser.add_argument("label", type=str, help="Label for the sign (no spaces)")
parser.add_argument("sequences", type=int, help="Number of sequences to record")
args = parser.parse_args()

label = args.label
no_sequences = args.sequences
sequence_length = 30

# ----------------------------
# Setup folders
# ----------------------------
DATA_PATH = os.path.join('data')
label_path = os.path.join(DATA_PATH, label)
os.makedirs(label_path, exist_ok=True)

# ----------------------------
# Init Mediapipe and webcam
# ----------------------------
mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for sequence in range(no_sequences):
        frames = []
        print(f"🔴 Recording '{label}' | Sequence {sequence}")
        time.sleep(2)

        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)

            keypoints = extract_keypoints(results)

            # Optional: normalize keypoints here
            keypoints = (keypoints - np.mean(keypoints)) / (np.std(keypoints) + 1e-8)
            frames.append(keypoints)

            # Display window
            cv2.namedWindow("Recording", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Recording", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Recording", image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Save the recorded sequence
        np.save(os.path.join(label_path, f"{sequence}.npy"), frames)
        print(f"✅ Saved: {label}/{sequence}.npy")

cap.release()
cv2.destroyAllWindows()

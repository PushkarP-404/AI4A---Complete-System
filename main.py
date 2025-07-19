import cv2
import os
import pyttsx3
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model  # type: ignore
from utils.mediapipe_utils import mediapipe_detection, extract_keypoints, draw_landmarks

# Parameters
sequence = []
sentence = []
predictions = []
threshold = 0.8
sequence_length = 30

# Load trained model and labels
model = load_model('sign_model.h5')
label_map = np.load('label_map.npy', allow_pickle=True).item()
inv_label_map = {v: k for k, v in label_map.items()}

# Init
mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)
engine = pyttsx3.init()  # Text-to-speech engine

with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:]

        if len(sequence) == sequence_length:
            input_data = np.expand_dims(sequence, axis=0)
            input_data = (input_data - np.mean(input_data)) / (np.std(input_data) + 1e-8)
            res = model.predict(input_data)[0]
            prediction = np.argmax(res)

            if res[prediction] > threshold:
                predicted_word = inv_label_map[prediction]
                if len(sentence) == 0 or predicted_word != sentence[-1]:
                    sentence.append(predicted_word)
                    engine.say(predicted_word)
                    engine.runAndWait()

        # Display result
        cv2.rectangle(image, (0, 0), (640, 40), (0, 0, 0), -1)
        cv2.putText(image, ' '.join(sentence[-5:]), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        print("Prediction:", ' '.join(sentence[-5:]))

        cv2.namedWindow("Sign Detection", cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty("Sign Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Sign Detection", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
# **** latest model with 4 words is stored in utils folder as backup  ****
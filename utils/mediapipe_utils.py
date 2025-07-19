import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Holistic once
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    """
    Performs detection on a single image frame using a given MediaPipe model.
    Returns the processed image and the detection results.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image.flags.writeable = False                   # Improve performance
    results = model.process(image)                  # Make prediction
    image.flags.writeable = True                    # Set back to writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR
    return image, results


def extract_keypoints(results):
    """
    Extracts hand keypoints from MediaPipe results.
    Returns a flattened array of [LEFT_HAND + RIGHT_HAND] keypoints (3D).
    Pads with zeros if any hand is missing.
    """
    lh = np.zeros(63)  # 21 landmarks × (x, y, z)
    rh = np.zeros(63)

    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()

    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()

    return np.concatenate([lh, rh])


def draw_landmarks(image, results):
    """
    Draw only hand landmarks on the image using MediaPipe drawing utils.
    Useful for clean display without face or body overlays.
    """
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    return image

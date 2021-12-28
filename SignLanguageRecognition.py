import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # bgr to rgb conversion
    image.flags.writeable = False  # set image to not writableqqq
    results = model.process(image)  # process sign language prediction
    image.flags.writeable = True  # set image to writable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # rgb to bgr conversion
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1,
                                                     circle_radius=1))  # draw face landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 21, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2,
                                                     circle_radius=2))  # draw pose landmarks
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=2,
                                                     circle_radius=2))  # draw left-hand landmarks
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=2,
                                                     circle_radius=2))  # draw right-hand landmarks


def extract_keypoints(results):
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, face, lh, rh])


DATA_PATH = os.path.join('MP_Data')  # Path for exported data, numpy arrays
actions = np.array(['hello', 'thanks', 'iloveyou'])  # Actions that we try to detect
no_sequences = 30  # Thirty videos worth of data
sequence_length = 30  # Videos are going to be 30 frames in length
start_folder = 30  # Folder start

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Loop through all actions (signs)
    for action in actions:
        # Loop through all sequences (videos)
        for sequence in range(no_sequences):
            # Loop through sequence length (frames)
            for frame_num in range(sequence_length):
                # Read camera feed
                ret, frame = cap.read()
                # Create detections
                image, results = mediapipe_detection(frame, holistic)
                # Draw all landmarks
                draw_landmarks(image, results)
                # wait 2 sec for next sequence
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('webcam_feed', image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('webcam_feed', image)

                # Export keypoint data
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                # check the ACTION directory does not exist
                if not(os.path.exists(os.path.join(DATA_PATH, action))):
                    # create the directory
                    os.mkdir(os.path.join(DATA_PATH, action))
                # check the SEQUENCE directory does not exist
                if not(os.path.exists(os.path.join(DATA_PATH, action, str(sequence)))):
                    # create the directory
                    os.mkdir(os.path.join(DATA_PATH, action, str(sequence)))
                # write the file
                np.save(npy_path, keypoints)
                # Break loop if user press q
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    cap.release()
    cv2.destroyAllWindows()

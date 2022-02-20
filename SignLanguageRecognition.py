import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split  # creates training partitions
from tensorflow.keras.utils import to_categorical  # covert data into encoded
from tensorflow.keras.models import Sequential  # to create  a sequential nueral network
from tensorflow.keras.layers import LSTM, Dense  # LSTM component to build model, allows to use action detection
from tensorflow.keras.callbacks import TensorBoard  # for logging and tracking
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score  # for evaluation

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

DATA_PATH = os.path.join('MP_Data')  # Path for exported data, numpy arrays
actions = np.array(['hello', 'thanks', 'iloveyou'])  # Actions that we try to detect
no_sequences = 30  # Thirty videos worth of data
sequence_length = 30  # Videos are going to be 30 frames in length
start_folder = 30  # Folder start


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # bgr to rgb conversion
    image.flags.writeable = False  # set image to not writable
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


def collect_keypoints():
    ##### Collect Keypoint Sequences
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:  # Loop through all actions (signs)
            for sequence in range(no_sequences):  # Loop through all sequences (videos)
                for frame_num in range(sequence_length):  # Loop through sequence length (frames)
                    ret, frame = cap.read()  # Read camera feed
                    image, results = mediapipe_detection(frame, holistic)  # Create detections
                    draw_landmarks(image, results)  # Draw all landmarks
                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {} of 30'.format(action, sequence),
                                    (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('webcam_feed', image)  # Show to screen
                        cv2.waitKey(2000)  # wait 2 sec for next sequence
                    else:
                        cv2.putText(image, 'Collecting frames for {} Video Number {} of 30'.format(action, sequence),
                                    (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('webcam_feed', image)  # Show to screen
                    keypoints = extract_keypoints(results)  # Export keypoint data
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    if not (os.path.exists(os.path.join(DATA_PATH, action))):  # check the ACTION directory does not exist
                        os.mkdir(os.path.join(DATA_PATH, action))  # create the directory
                    if not (os.path.exists(os.path.join(DATA_PATH, action, str(sequence)))):  # check the SEQUENCE directory does not exist
                        os.mkdir(os.path.join(DATA_PATH, action, str(sequence)))  # create the directory
                    np.save(npy_path, keypoints)  # write the file
                    if cv2.waitKey(10) & 0xFF == ord('q'):  # Break loop if user press q
                        break
        cap.release()
        cv2.destroyAllWindows()
    return


def train_lstm_network():
    ##### Pre-process data and create labels
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []  # 2 bank arrays, sequences for feature data, labels for labels
    for action in actions:  # loop through all actions
        for sequence in range(no_sequences):  # loop through all sequences
            window = []  # to store all frames for a sequence
            for frame_num in range(sequence_length):  # loop through all frames
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))  # load frames
                window.append(res)  # add to window array
            sequences.append(window)  # add the video to the sequences array
            labels.append(label_map[action])  # add sequence to labels array
    X = np.array(sequences)
    Y = to_categorical(labels).astype(int)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)  # setting up training variables

    ##### Building and training LTSM Nueral network model
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)  # variable for monitoring the learning process

    model = Sequential()  # instantiating the sequence (sequential api)
    # create LSTM layers
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    # create Dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))  # actions layer

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])  # Compile neural model
    model.fit(X_train, Y_train, epochs=300, callbacks=[tb_callback])  # fit model
    model.save('action.h5')
    return model


#### Evaluation (confusion matrix and accuracy)
#yhat = model.predict(X_test)  # predict values
#ytrue = np.argmax(Y_test, axis=1).tolist()
#yhat = np.argmax(yhat, axis=1).tolist()
#multilabel_confusion_matrix(ytrue, yhat)
#accuracy_score(ytrue, yhat)


def activate_slr(model):
    # Detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.7

    #### SLR
    cap = cv2.VideoCapture(0)  # initialize the camera
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:  # Set mediapipe model
        while cap.isOpened():  # While cam is opened
            ret, frame = cap.read()  # Read frame
            image, results = mediapipe_detection(frame, holistic)  # Make detections
            draw_landmarks(image, results)  # draw landmarks

            #### Prediction logic
            keypoints = extract_keypoints(results)  # Extract key points from detections
            sequence.append(keypoints)  # append keypoints to sequence
            sequence = sequence[-30:]  # Grab last 30 frames
            #model.load_weights('action.h5')  # Load trained model

            if len(sequence) == 30:  # if the length of the sequence is 30
                res = model.predict(np.expand_dims(sequence, axis=0))[0]  # run prediction for 1 sequence
                predictions.append(np.argmax(res))  # append all predictions

                #### Visualization logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):  # check that prediction is the same in last 10 frames
                    if res[np.argmax(res)] > threshold:  # check if result is above threshold
                        if len(sentence) > 0:  # check that sentence is not empty
                            if actions[np.argmax(res)] != sentence[-1]:  # check that current detection is not the same as last detection
                                sentence.append(actions[np.argmax(res)])  # append sentence
                        else:
                            sentence.append(actions[np.argmax(res)])  # append sentence  # append sentence

                if len(sentence) > 5:  # if sentence is greater than 5 words
                    sentence = sentence[-5:]  # grab last 5 values

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)  # specify rectangle
            cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # Render sentence with space
            cv2.imshow('webcam_feed', image)  # Show to screen

            #### Break logic
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    return


#### Menu
#collect_keypoints()
trained_model = train_lstm_network()
activate_slr(trained_model)


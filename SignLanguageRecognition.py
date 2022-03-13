import os
import sys
import time
import cv2
import mediapipe as mp
import numpy as np
import pygame  # Used to play sounds
from PyQt5.QtGui import QImage, QPixmap, QColor  # UI framework
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QObject  # UI framework
from PyQt5.uic import loadUi  # UI framework
from PyQt5 import QtWidgets  # UI framework
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton,  QMessageBox  # UI framework
from PyQt5.QtCore import QTimer  # Timer
from gtts import gTTS  # Google translate API
from sklearn.model_selection import train_test_split  # creates training partitions
from tensorflow.keras.utils import to_categorical  # covert data into encoded
from tensorflow.keras.models import Sequential  # to create  a sequential nueral network
from tensorflow.keras.layers import LSTM, Dense  # LSTM component to build model, allows to use action detection
from tensorflow.keras.callbacks import TensorBoard  # for logging and tracking
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
DATA_PATH = os.path.join('MP_Data')  # Path for exported data, numpy arrays
no_sequences = 30  # Thirty videos worth of data
sequence_length = 30  # Videos are going to be 30 frames in length
start_folder = 30  # Folder start
language = 'en'  # Language for text to speech


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


def train_lstm_network(trainactions):
    # Pre-process data and create labels
    actions = np.array(trainactions)  # Actions to detect
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []  # 2 bank arrays, sequences for feature data, labels for labels
    for action in actions:  # loop through all actions
        for sequence in range(no_sequences):  # loop through all sequences
            window = []  # to store all frames for a sequence
            for frame_num in range(sequence_length):  # loop through all frames
                res = np.load(os.path.join('SignLanguages', choosevocab.languageSel, 'MP_Data', action, str(sequence), "{}.npy".format(frame_num)))  # load frames
                window.append(res)  # add to window array
            sequences.append(window)  # add the video to the sequences array
            labels.append(label_map[action])  # add sequence to labels array
    X = np.array(sequences)
    Y = to_categorical(labels).astype(int)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)  # setting up training variables

    # Building and training LTSM Nueral network model
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

    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])  # Compile neural model
    model.fit(X_train, Y_train, epochs=300, callbacks=[tb_callback])  # fit model
    model_name = os.path.join('Models', choosevocab.languageSel)
    model.save(model_name)
    teachUI.Loadingwidget.setVisible(False)
    return


class playAudioFile(QThread):
    finished = pyqtSignal()

    def run(self):
        pygame.mixer.init()
        pygame.mixer.music.load(os.path.join('UtterMP3', "prediction.mp3"))
        pygame.mixer.music.play()
        time.sleep(1)
        pygame.mixer.music.unload()
        self.finished.emit()


# def collect_keypoints():
#     ##### Collect Keypoint Sequences
#     cap = cv2.VideoCapture(0)
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         for action in actions:  # Loop through all actions (signs)
#             for sequence in range(no_sequences):  # Loop through all sequences (videos)
#                 for frame_num in range(sequence_length):  # Loop through sequence length (frames)
#                     ret, frame = cap.read()  # Read camera feed
#                     image, results = mediapipe_detection(frame, holistic)  # Create detections
#                     draw_landmarks(image, results)  # Draw all landmarks
#                     if frame_num == 0:
#                         cv2.putText(image, 'STARTING COLLECTION', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
#                         cv2.putText(image, 'Collecting frames for {} Video Number {} of 30'.format(action, sequence),
#                                     (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#                         cv2.imshow('webcam_feed', image)  # Show to screen
#                         cv2.waitKey(2000)  # wait 2 sec for next sequence
#                     else:
#                         cv2.putText(image, 'Collecting frames for {} Video Number {} of 30'.format(action, sequence),
#                                     (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#                         cv2.imshow('webcam_feed', image)  # Show to screen
#                     keypoints = extract_keypoints(results)  # Export keypoint data
#                     npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
#                     if not (os.path.exists(os.path.join(DATA_PATH, action))):  # check the ACTION directory does not exist
#                         os.mkdir(os.path.join(DATA_PATH, action))  # create the directory
#                     if not (os.path.exists(os.path.join(DATA_PATH, action, str(sequence)))):  # check the SEQUENCE directory does not exist
#                         os.mkdir(os.path.join(DATA_PATH, action, str(sequence)))  # create the directory
#                     np.save(npy_path, keypoints)  # write the file
#                     if cv2.waitKey(10) & 0xFF == ord('q'):  # Break loop if user press q
#                         break
#         cap.release()
#         cv2.destroyAllWindows()
#     return


# Evaluation (confusion matrix and accuracy)
# yhat = model.predict(X_test)  # predict values
# ytrue = np.argmax(Y_test, axis=1).tolist()
# yhat = np.argmax(yhat, axis=1).tolist()
# multilabel_confusion_matrix(ytrue, yhat)
# accuracy_score(ytrue, yhat)
# def activate_slr(model):
#
#     # Detection variables
#     sequence = []
#     sentence = []
#     predictions = []
#     threshold = 0.7
#
#     #### SLR
#     cap = cv2.VideoCapture(0)  # initialize the camera
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:  # Set mediapipe model
#         while cap.isOpened():  # While cam is opened
#             ret, frame = cap.read()  # Read frame
#             image, results = mediapipe_detection(frame, holistic)  # Make detections
#             draw_landmarks(image, results)  # draw landmarks
#
#             #### Prediction logic
#             keypoints = extract_keypoints(results)  # Extract key points from detections
#             sequence.append(keypoints)  # append keypoints to sequence
#             sequence = sequence[-30:]  # Grab last 30 frames
#             #model.load_weights('action.h5')  # Load trained model
#
#             if len(sequence) == 30:  # if the length of the sequence is 30
#                 res = model.predict(np.expand_dims(sequence, axis=0))[0]  # run prediction for 1 sequence
#                 predictions.append(np.argmax(res))  # append all predictions
#
#                 #### Visualization logic
#                 if np.unique(predictions[-10:])[0] == np.argmax(res):  # check that prediction is the same in last 10 frames
#                     if res[np.argmax(res)] > threshold:  # check if result is above threshold
#                         if len(sentence) > 0:  # check that sentence is not empty
#                             if actions[np.argmax(res)] != sentence[-1]:  # check that current detection is not the same as last detection
#                                 sentence.append(actions[np.argmax(res)])  # append sentence
#                         else:
#                             sentence.append(actions[np.argmax(res)])  # append sentence  # append sentence
#
#                 if len(sentence) > 5:  # if sentence is greater than 5 words
#                     sentence = sentence[-5:]  # grab last 5 values
#
#             cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)  # specify rectangle
#             cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # Render sentence with space
#             cv2.imshow('webcam_feed', image)  # Show to screen
#
#             #### Break logic
#             if cv2.waitKey(10) & 0xFF == ord('q'):
#                 break
#         cap.release()
#         cv2.destroyAllWindows()
#     return


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi(os.path.join('UIpages', "Main.ui"), self)
        self.teachvocButton.clicked.connect(self.gotoTeachNewVocWindow)
        self.activateSLRButton.clicked.connect(self.ChooseLanguageSLRWindow)
        self.exitButton.clicked.connect(self.ExitApp)

    def gotoTeachNewVocWindow(self):
        widget.setCurrentIndex(1)

    def ChooseLanguageSLRWindow(self):
        widget.setCurrentIndex(8)

    def ExitApp(self):
        QApplication.quit()


class TeachNewVocWindow(QMainWindow):
    def __init__(self):
        super(TeachNewVocWindow, self).__init__()
        loadUi(os.path.join('UIpages', "TeachNewVoc.ui"), self)
        self.backButton.clicked.connect(self.gotoMainWindow)
        self.addLangButton.clicked.connect(self.gotoAddLanguageWindow)
        self.languageTable.setColumnWidth(0, 220)
        self.languageTable.setColumnWidth(1, 200)
        self.languageTable.setColumnWidth(2, 150)
        self.languageTable.setColumnWidth(3, 100)

        self.editbtn = QPushButton(self.languageTable)
        self.editbtn.setText('edit')
        self.editbtn.clicked.connect(self.gotoEditLanguageWindow)

        self.languageTable.cellDoubleClicked.connect(self.gotoChooseVocabularyWindow)

        self.timer = QTimer()
        self.timer.timeout.connect(self.loadData)
        self.timer.setInterval(2000)
        self.timer.start()

    def loadData(self):
        languages = []
        files = os.listdir(os.path.join('SignLanguages'))
        for filename in files:
            if os.path.exists(os.path.join('SignLanguages', filename, 'MP_Data')):
                data = os.listdir(os.path.join('SignLanguages', filename, 'MP_Data'))
                totalVoc = 0
                for vocabulary in data:
                    totalVoc += 1
            else:
                totalVoc = 0
            languages.append({"Language": filename, "Vocabulary Amount": str(totalVoc), "Created By": "Alexei"})

        row = 0
        self.languageTable.setRowCount(len(languages))
        for language in languages:
            self.languageTable.setItem(row, 0, QtWidgets.QTableWidgetItem(language["Language"]))
            self.languageTable.setItem(row, 1, QtWidgets.QTableWidgetItem(language["Vocabulary Amount"]))
            self.languageTable.setItem(row, 2, QtWidgets.QTableWidgetItem(language["Created By"]))
            self.languageTable.setCellWidget(row, 3, self.editbtn)
            row = row+1

    def gotoMainWindow(self):
        widget.setCurrentIndex(0)

    def gotoAddLanguageWindow(self):
        widget.setCurrentIndex(2)

    def gotoEditLanguageWindow(self):
        widget.setCurrentIndex(3)

    def gotoChooseVocabularyWindow(self, row, column):
        item = self.languageTable.item(row, 0)
        choosevocab.languageSel = item.text()
        widget.setCurrentIndex(4)


class AddLanguageWindow(QMainWindow):
    def __init__(self):
        super(AddLanguageWindow, self).__init__()
        loadUi(os.path.join('UIpages', "insertLanguage.ui"), self)
        self.backButton.clicked.connect(self.gotoTeachNewVocWindow)
        self.addButton.clicked.connect(self.addLanguage)

    def addLanguage(self):
        input = self.languageInput.text()
        if not os.path.exists(os.path.join('SignLanguages', input, 'MP_Data')):
            os.mkdir(os.path.join('SignLanguages', input))
            os.mkdir(os.path.join('SignLanguages', input, 'MP_Data'))
        else:
            QMessageBox.critical(self, "Oops..", "It seems the language has already been created")
        widget.setCurrentIndex(1)

    def gotoTeachNewVocWindow(self):
        widget.setCurrentIndex(1)


class EditLanguageWindow(QMainWindow):
    def __init__(self):
        super(EditLanguageWindow, self).__init__()
        loadUi(os.path.join('UIpages', "editLanguage.ui"), self)
        self.backButton.clicked.connect(self.gotoTeachNewVocWindow)
        self.saveButton.clicked.connect(self.editLanguage)

    def editLanguage(self):
        input = self.languageInput.text()
        if not os.path.exists(os.path.join('SignLanguages', input, 'MP_Data')):
            os.mkdir(os.path.join('SignLanguages', input))
            os.mkdir(os.path.join('SignLanguages', input, 'MP_Data'))
        widget.setCurrentIndex(1)

    def gotoTeachNewVocWindow(self):
        widget.setCurrentIndex(1)


class ChooseVocabularyWindow(QMainWindow):
    def __init__(self):
        super(ChooseVocabularyWindow, self).__init__()
        loadUi(os.path.join('UIpages', "chooseVocabulary.ui"), self)
        self.backButton.clicked.connect(self.gotoTeachNewVocWindow)
        self.addVocButton.clicked.connect(self.gotoAddVocabularyWindow)
        self.vocabularyTable.setColumnWidth(0, 220)
        self.vocabularyTable.setColumnWidth(1, 200)
        self.vocabularyTable.setColumnWidth(2, 150)
        self.vocabularyTable.setColumnWidth(3, 100)

        self.languageSel = ''

        self.editbtn = QPushButton(self.vocabularyTable)
        self.editbtn.setText('edit')
        self.editbtn.clicked.connect(self.gotoEditVocabularyWindow)

        self.vocabularyTable.cellDoubleClicked.connect(self.gotoTeachRecogWindow)

        self.timer = QTimer()
        self.timer.timeout.connect(self.loadDatavoc)
        self.timer.setInterval(2000)
        self.timer.start()

    def loadDatavoc(self):
        self.languagelabeltext = "Language: " + str(self.languageSel)
        self.languagelabel.setText(self.languagelabeltext)
        languages = []
        if os.path.exists(os.path.join('SignLanguages', self.languageSel, 'MP_Data')):
            files = os.listdir(os.path.join('SignLanguages', self.languageSel, 'MP_Data'))
            for filename in files:
                if os.path.exists(os.path.join('SignLanguages', self.languageSel, 'MP_Data', filename)):
                    data = os.listdir(os.path.join('SignLanguages', self.languageSel, 'MP_Data', filename))
                    totalVoc = 0
                    for vocabulary in data:
                        totalVoc += 1
                else:
                    totalVoc = 0
                languages.append({"Vocabulary": filename, "Vocabulary Amount": str(totalVoc), "Created By": "Alexei"})

        row = 0
        self.vocabularyTable.setRowCount(len(languages))
        for language in languages:
            self.vocabularyTable.setItem(row, 0, QtWidgets.QTableWidgetItem(language["Vocabulary"]))
            self.vocabularyTable.setItem(row, 1, QtWidgets.QTableWidgetItem(language["Vocabulary Amount"]))
            self.vocabularyTable.setItem(row, 2, QtWidgets.QTableWidgetItem(language["Created By"]))
            self.vocabularyTable.setCellWidget(row, 3, self.editbtn)
            row = row + 1

    def gotoTeachNewVocWindow(self):
        widget.setCurrentIndex(1)

    def gotoAddVocabularyWindow(self):
        widget.setCurrentIndex(5)

    def gotoEditVocabularyWindow(self):
        widget.setCurrentIndex(6)

    def gotoTeachRecogWindow(self, row, column):
        item = self.vocabularyTable.item(row, 0)
        chosenVoc = item.text()
        data = os.listdir(os.path.join('SignLanguages', self.languageSel, 'MP_Data', chosenVoc))
        totalVoc = 0
        for vocabulary in data:
            totalVoc += 1
        if totalVoc > 0:
            QMessageBox.critical(self, "Already exists",
                                 "After the data is gathered, it will overwrite existing data for this vocabulary")
        teachUI.vocabularySel = chosenVoc
        widget.setCurrentIndex(7)
        teachUI.threadState = True


class AddVocabularyWindow(QMainWindow):
    def __init__(self):
        super(AddVocabularyWindow, self).__init__()
        loadUi(os.path.join('UIpages', "insertVocabulary.ui"), self)
        self.backButton.clicked.connect(self.gotoChooseVocabularyWindow)
        self.addButton.clicked.connect(self.addVocabulary)

    def addVocabulary(self):
        input = self.vocabularyInput.text()
        language = choosevocab.languageSel
        if not os.path.exists(os.path.join('SignLanguages', language, 'MP_Data', input)):
            os.mkdir(os.path.join('SignLanguages', language, 'MP_Data', input))
        else:
            QMessageBox.critical(self, "Oops..", "It seems the vocabulary has already been created")
        widget.setCurrentIndex(4)

    def gotoChooseVocabularyWindow(self):
        widget.setCurrentIndex(4)


class ProgressBarThread(QThread):
    PBValueSig = pyqtSignal(int)

    def run(self):
        i = 0
        while i <= 31:
            i = i + 1
            value = (i/31) * 100
            self.PBValueSig.emit(value)
            time.sleep(1)


class EditVocabularyWindow(QMainWindow):
    def __init__(self):
        super(EditVocabularyWindow, self).__init__()
        loadUi(os.path.join('UIpages', "editVocabulary.ui"), self)
        self.backButton.clicked.connect(self.gotoChooseVocabularyWindow)
        self.saveButton.clicked.connect(self.editVocabulary)

    def editVocabulary(self):
        input = self.editvocabularyInput.text()
        language = choosevocab.languageSel
        if not os.path.exists(os.path.join('SignLanguages', language, 'MP_Data', input)):
            os.mkdir(os.path.join('SignLanguages', language, 'MP_Data', input))
        widget.setCurrentIndex(4)

    def gotoChooseVocabularyWindow(self):
        widget.setCurrentIndex(4)


class TeachingThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        ##### Collect Keypoint Sequences
        teachUI.Loadingwidget.setVisible(False)
        cap = cv2.VideoCapture(0)
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            action = teachUI.vocabularySel
            for sequence in range(no_sequences):  # Loop through all sequences (videos)
                for frame_num in range(sequence_length):  # Loop through sequence length (frames)
                    ret, frame = cap.read()  # Read camera feed
                    image, results = mediapipe_detection(frame, holistic)  # Create detections
                    draw_landmarks(image, results)  # Draw all landmarks
                    if frame_num == 0:
                        teachUI.label_sequence.setText('Collecting frames for: "{}", Video Number: {} of 30'.format(action, sequence))
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgbImage.shape
                        bytesPerLine = ch * w
                        convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                        p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                        cv2.waitKey(2000)  # wait 2 sec for next sequence
                    else:
                        teachUI.label_sequence.setText('Collecting frames for: "{}", Video Number: {} of 30'.format(action, sequence))
                        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgbImage.shape
                        bytesPerLine = ch * w
                        convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                        p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                    keypoints = extract_keypoints(results)  # Export keypoint data
                    npy_path = os.path.join('SignLanguages', choosevocab.languageSel, 'MP_Data', action, str(sequence), str(frame_num))
                    if not (os.path.exists(os.path.join('SignLanguages', choosevocab.languageSel, 'MP_Data', action))):  # check the ACTION directory does not exist
                        os.mkdir(os.path.join('SignLanguages', choosevocab.languageSel, 'MP_Data', action))  # create the directory
                    if not (os.path.exists(os.path.join('SignLanguages', choosevocab.languageSel, 'MP_Data', action,  str(sequence)))):  # check the SEQUENCE directory does not exist
                        os.mkdir(os.path.join('SignLanguages', choosevocab.languageSel, 'MP_Data', action, str(sequence)))  # create the directory
                    np.save(npy_path, keypoints)  # write the file
                    self.changePixmap.emit(p)
            teachUI.label_sequence.setText('Please wait...')
            teachUI.Loadingwidget.setVisible(True)
            teachUI.progressbarth.start()
            files = os.listdir(os.path.join('SignLanguages', (choosevocab.languageSel), 'MP_Data'))
            train_lstm_network(files)
            cap.release()
            cv2.destroyAllWindows()
            widget.setCurrentIndex(4)
            teachUI.threadState = False
            teachUI.progressbarth.terminate()
        return


class TeachRecogWindow(QMainWindow):
    def __init__(self):
        super(TeachRecogWindow, self).__init__()
        loadUi(os.path.join('UIpages', "teachUI.ui"), self)
        self.backButton.clicked.connect(self.gotoChooseVocabularyWindow)

        grey = QPixmap(731, 361)  # create a grey pixmap
        grey.fill(QColor('darkGray'))
        self.image_label.setPixmap(grey)  # set the image to the grey pixmap

        self.vocabularySel = ""
        self.threadState = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.threadControl)
        self.timer.setInterval(1000)
        self.timer.start()

        self.th = TeachingThread(self)
        self.th.changePixmap.connect(self.setImage)

        self.progressbarth = ProgressBarThread(self)
        self.progressbarth.PBValueSig.connect(self.updateProgressBar)

    @pyqtSlot(int)
    def updateProgressBar(self, value):
        self.progressBar.setValue(value)

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.image_label.setPixmap(QPixmap.fromImage(image))

    def threadControl(self):
        if (self.threadState is True) & (self.th.isRunning() is False):
            print("starting")
            self.th.start()
        elif (self.threadState is False) & (self.th.isRunning() is True):
            print("terminating")
            self.th.terminate()

    def gotoChooseVocabularyWindow(self):
        widget.setCurrentIndex(4)
        self.threadState = False


class ChooseLanguageSLRWindow(QMainWindow):
    def __init__(self):
        super(ChooseLanguageSLRWindow, self).__init__()
        loadUi(os.path.join('UIpages', "chooseLanguageSLR.ui"), self)
        self.backButton.clicked.connect(self.gotoMainWindow)
        self.languageTable.setColumnWidth(0, 220)
        self.languageTable.setColumnWidth(1, 200)
        self.languageTable.setColumnWidth(2, 150)

        self.languageTable.cellDoubleClicked.connect(self.gotoslrUI)

        self.timer = QTimer()
        self.timer.timeout.connect(self.loadData)
        self.timer.setInterval(2000)
        self.timer.start()

    def loadData(self):
        languages = []
        files = os.listdir(os.path.join('SignLanguages'))
        for filename in files:
            if os.path.exists(os.path.join('SignLanguages', filename, 'MP_Data')):
                data = os.listdir(os.path.join('SignLanguages', filename, 'MP_Data'))
                totalVoc = 0
                for vocabulary in data:
                    totalVoc += 1
            else:
                totalVoc = 0
            if os.path.exists(os.path.join('Models', filename)):
                languages.append({"Language": filename, "Vocabulary Amount": str(totalVoc), "Created By": "Alexei"})

            row = 0
            self.languageTable.setRowCount(len(languages))
            for language in languages:
                self.languageTable.setItem(row, 0, QtWidgets.QTableWidgetItem(language["Language"]))
                self.languageTable.setItem(row, 1, QtWidgets.QTableWidgetItem(language["Vocabulary Amount"]))
                self.languageTable.setItem(row, 2, QtWidgets.QTableWidgetItem(language["Created By"]))
                row = row+1

    def gotoMainWindow(self):
        widget.setCurrentIndex(0)

    def gotoslrUI(self, row, column):
        item = self.languageTable.item(row, 0)
        slrUI.languageSel = item.text()
        widget.setCurrentIndex(9)
        slrUI.threadState = True


class SlrThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        # Detection variables
        sequence = []
        sentence = []
        predictions = []
        threshold = 0.7
        slractions = os.listdir(os.path.join('SignLanguages', (slrUI.languageSel), 'MP_Data'))
        actions = np.array(slractions)  # Actions to detect

        #### SLR
        cap = cv2.VideoCapture(0)  # initialize the camera
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:  # Set mediapipe model
            model = keras.models.load_model(os.path.join('Models', slrUI.languageSel))  # Load trained model
            while cap.isOpened():  # While cam is opened
                ret, frame = cap.read()  # Read frame
                image, results = mediapipe_detection(frame, holistic)  # Make detections
                draw_landmarks(image, results)  # draw landmarks

                #### Prediction logic
                keypoints = extract_keypoints(results)  # Extract key points from detections
                sequence.append(keypoints)  # append keypoints to sequence
                sequence = sequence[-30:]  # Grab last 30 frames

                if len(sequence) == 30:  # if the length of the sequence is 30
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]  # run prediction for 1 sequence
                    predictions.append(np.argmax(res))  # append all predictions

                    #### Visualization logic
                    if np.unique(predictions[-10:])[0] == np.argmax(
                            res):  # check that prediction is the same in last 10 frames
                        if res[np.argmax(res)] > threshold:  # check if result is above threshold
                            if len(sentence) > 0:  # check that sentence is not empty
                                if actions[np.argmax(res)] != sentence[-1]:  # check that current detection is not the same as last detection
                                    sentence.append(actions[np.argmax(res)])  # append sentence
                                    pred = gTTS(text=actions[np.argmax(res)], lang=language, slow=False)
                                    pred.save(os.path.join('UtterMP3', "prediction.mp3"))  # save voice line of prediction
                                    self.thplay = playAudioFile(self)
                                    self.thplay.finished.connect(self.thplay.quit)
                                    self.thplay.start()
                            else:
                                sentence.append(actions[np.argmax(res)])  # append sentence  # append sentence
                    if len(sentence) > 5:  # if sentence is greater than 5 words
                        sentence = sentence[-5:]  # grab last 5 values
                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)  # specify rectangle
                slrUI.label_prediction.setText(' '.join(sentence))
                rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
            slrUI.threadState = False
            cap.release()
            cv2.destroyAllWindows()
        return


class SlrWindow(QMainWindow):
    def __init__(self):
        super(SlrWindow, self).__init__()
        loadUi(os.path.join('UIpages', "slrUI.ui"), self)
        self.backButton.clicked.connect(self.gotoChooseLanguageSLRWindow)

        grey = QPixmap(731, 361)  # create a grey pixmap
        grey.fill(QColor('darkGray'))
        self.image_label.setPixmap(grey)  # set the image to the grey pixmap

        self.languageSel = ""
        self.threadState = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.threadControl)
        self.timer.setInterval(1000)
        self.timer.start()

        self.thslr = SlrThread(self)
        self.thslr.changePixmap.connect(self.setImage)

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.image_label.setPixmap(QPixmap.fromImage(image))

    def threadControl(self):
        if (self.threadState is True) & (self.thslr.isRunning() is False):
            print("starting")
            self.thslr.start()
        elif (self.threadState is False) & (self.thslr.isRunning() is True):
            print("terminating")
            self.thslr.terminate()

    def gotoChooseLanguageSLRWindow(self):
        widget.setCurrentIndex(8)
        self.threadState = False


app = QApplication(sys.argv)
widget = QtWidgets.QStackedWidget()

mainwindow = MainWindow()
teachnewvoc = TeachNewVocWindow()
addnewlang = AddLanguageWindow()
editnewlang = EditLanguageWindow()
choosevocab = ChooseVocabularyWindow()
addnewvoc = AddVocabularyWindow()
editnewvoc = EditVocabularyWindow()
teachUI = TeachRecogWindow()
chooselangSLR = ChooseLanguageSLRWindow()
slrUI = SlrWindow()

widget.addWidget(mainwindow)
widget.addWidget(teachnewvoc)
widget.addWidget(addnewlang)
widget.addWidget(editnewlang)
widget.addWidget(choosevocab)
widget.addWidget(addnewvoc)
widget.addWidget(editnewvoc)
widget.addWidget(teachUI)
widget.addWidget(chooselangSLR)
widget.addWidget(slrUI)

widget.setFixedWidth(829)
widget.setFixedHeight(546)
widget.show()

try:
    sys.exit(app.exec_())
except:
    print("Exiting")

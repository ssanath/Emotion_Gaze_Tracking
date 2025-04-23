import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class EmotionDetector:
    def __init__(self, weights_path="model.h5", haar_path="haarcascade_frontalface_default.xml"):
        self.model = self.build_model()
        self.model.load_weights(weights_path)
        self.face_cascade = cv2.CascadeClassifier(haar_path)
        self.emotion_dict = {
            0: "Angry", 1: "Disgusted", 2: "Fearful",
            3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
        }

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))
        return model

    def detect_emotion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            return "No Face", 0.0

        (x, y, w, h) = faces[0]  # Detect only the first face
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = cv2.resize(roi_gray, (48, 48))
        cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)
        cropped_img = cropped_img.astype("float32") / 255.0

        prediction = self.model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        confidence = float(prediction[0][maxindex])

        return self.emotion_dict[maxindex], confidence

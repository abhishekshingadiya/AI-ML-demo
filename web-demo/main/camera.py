import os

import cv2
import numpy as np
from keras.models import load_model
from pygame import mixer

mixer.init()


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.sound = mixer.Sound('./haar cascade files/audio/alarm.wav')

        self.face = cv2.CascadeClassifier('./haar cascade files/haarcascade_frontalface_alt.xml')
        self.leye = cv2.CascadeClassifier('./haar cascade files/haarcascade_lefteye_2splits.xml')
        self.reye = cv2.CascadeClassifier('./haar cascade files/haarcascade_righteye_2splits.xml')

        self.lbl = ['Close', 'Open']

        self.model = load_model('./models/cnnCat2.h5')
        self.path = os.getcwd()
        self.cap = cv2.VideoCapture(0)
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.count = 0
        self.score = 0
        self.thicc = 2
        self.rpred = [99]
        self.lpred = [99]

    def __del__(self):
        self.video.release()

    def get_frame1(self):
        ret, frame = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.

        height, width = frame.shape[:2]
        if self.count % 5 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
            left_eye = self.leye.detectMultiScale(gray)
            right_eye = self.reye.detectMultiScale(gray)

            cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

            for (x, y, w, h) in right_eye:
                r_eye = frame[y:y + h, x:x + w]
                self.count = self.count + 1
                r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
                r_eye = cv2.resize(r_eye, (24, 24))
                r_eye = r_eye / 255
                r_eye = r_eye.reshape(24, 24, -1)
                r_eye = np.expand_dims(r_eye, axis=0)
                self.rpred = np.argmax(self.model.predict(r_eye), axis=-1)
                if (self.rpred[0] == 1):
                    self.lbl = 'Open'
                if (self.rpred[0] == 0):
                    self.lbl = 'Closed'
                break

            for (x, y, w, h) in left_eye:
                l_eye = frame[y:y + h, x:x + w]
                self.count = self.count + 1
                l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
                l_eye = cv2.resize(l_eye, (24, 24))
                l_eye = l_eye / 255
                l_eye = l_eye.reshape(24, 24, -1)
                l_eye = np.expand_dims(l_eye, axis=0)
                self.lpred = np.argmax(self.model.predict(l_eye), axis=-1)
                if (self.lpred[0] == 1):
                    self.lbl = 'Open'
                if (self.lpred[0] == 0):
                    self.lbl = 'Closed'
                break

            if (self.rpred[0] == 0 and self.lpred[0] == 0):
                self.score = self.score + 1
                cv2.putText(frame, "Closed" , (10, height - 40), self.font, 1,
                            (255, 255, 255), 1, cv2.LINE_AA)
            # if(rpred[0]==1 or lpred[0]==1):
            else:
                self.score = self.score - 1
                cv2.putText(frame, "Open" , (10, height - 40), self.font, 1,
                            (255, 255, 255), 1, cv2.LINE_AA)

            if (self.score < 0):
                self.score = 0
                # cv2.putText(frame, 'Score:' + str(self.score), (100, height - 20), self.font, 1, (255, 255, 255), 1,
                #             cv2.LINE_AA)
            if (self.score > 100):
                # person is feeling sleepy so we beep the alarm
                cv2.imwrite(os.path.join(self.path, 'image.jpg'), frame)
                try:
                    self.sound.play()

                except:  # isplaying = False
                    pass
                if (self.thicc < 16):
                    thicc = self.thicc + 2
                else:
                    thicc = self.thicc - 2
                    if (thicc < 2):
                        thicc = 2
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

            # frame_flip = cv2.flip(frame, 1)
        cv2.putText(frame, 'Score:' + str(self.score), (10, height - 20), self.font, 1,
                    (255, 255, 255), 1, cv2.LINE_AA)
        ret, jpeg = cv2.imencode('.jpg', frame)

        self.count += 1
        return jpeg.tobytes()

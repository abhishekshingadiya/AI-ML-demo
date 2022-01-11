import cv2
import os
from keras.models import load_model
import numpy as np

import time



lbl = ['Close', 'Open']


path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]
count = 0
while (True):
    ret, frame = cap.read()
    if ret:
        height, width = frame.shape[:2]
        if count % 3 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



            cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)


            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
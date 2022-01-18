import io
import os
import traceback
from base64 import encodebytes

import cv2
import numpy as np
# from Spotipy import *
import pandas as pd
from PIL import Image
from flask import Flask
from flask import request
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential

from app_utils import clean_all
from app_utils import convertToJPG
from app_utils import create_directory
from app_utils import download
from app_utils import generate_random_filename

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
ds_factor = 0.6

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
music_dist = {0: "songs/angry.csv", 1: "songs/disgusted.csv ", 2: "songs/fearful.csv", 3: "songs/happy.csv",
              4: "songs/neutral.csv", 5: "songs/sad.csv", 6: "songs/surprised.csv"}

show_text = [0]

app = Flask(__name__)

headings = ("Name", "Album", "Artist")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def music_rec():
    # print('---------------- Value ------------', music_dist[show_text[0]])
    df = pd.read_csv(music_dist[show_text[0]])
    df = df[['Name', 'Album', 'Artist']]
    df = df.head(15)
    return df


def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r')  # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG')  # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')  # encode as base64
    return encoded_img


@app.route("/process", methods=["POST"])
def process_image():
    input_path = generate_random_filename(upload_directory, "jpeg")
    output_path = os.path.join(results_img_directory, os.path.basename(input_path))

    try:
        if 'file' in request.files:
            file = request.files['file']
            if allowed_file(file.filename):
                file.save(input_path)

        else:
            url = request.json["url"]
            download(url, input_path)

        result, df1 = None, music_rec().head(15)

        try:
            img = cv2.imread(input_path)
            image = cv2.resize(img, (600, 500))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
            df1 = pd.read_csv(music_dist[show_text[0]])
            df1 = df1[['Name', 'Album', 'Artist']]
            df1 = df1.head(15)
            for (x, y, w, h) in face_rects:
                cv2.rectangle(image, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 2)
                roi_gray_frame = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                prediction = emotion_model.predict(cropped_img)

                maxindex = int(np.argmax(prediction))
                show_text[0] = maxindex
                # print("===========================================",music_dist[show_text[0]],"===========================================")
                # print(df1)
                cv2.putText(image, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255),
                            2, cv2.LINE_AA)
                df1 = music_rec()

            last_frame1 = image.copy()
            pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(last_frame1)
            img = np.array(img)
            # ret, jpeg = cv2.imencode('.jpg', img)
            # return jpeg.tobytes(), df1
            # encode as a jpeg image and return it
            result, df1 = last_frame1, df1
        except:
            convertToJPG(input_path)
            img = cv2.imread(input_path)
            image = cv2.resize(img, (600, 500))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
            df1 = pd.read_csv(music_dist[show_text[0]])
            df1 = df1[['Name', 'Album', 'Artist']]
            df1 = df1.head(15)
            for (x, y, w, h) in face_rects:
                cv2.rectangle(image, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 2)
                roi_gray_frame = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                prediction = emotion_model.predict(cropped_img)

                maxindex = int(np.argmax(prediction))
                show_text[0] = maxindex
                # print("===========================================",music_dist[show_text[0]],"===========================================")
                # print(df1)
                cv2.putText(image, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255),
                            2, cv2.LINE_AA)
                df1 = music_rec()

            last_frame1 = image.copy()
            pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(last_frame1)
            img = np.array(img)
            # ret, jpeg = cv2.imencode('.jpg', img)
            # return jpeg.tobytes(), df1
            # encode as a jpeg image and return it
            result, df1 = last_frame1, df1

        finally:
            if result is not None:
                cv2.imwrite(output_path, result)
        encoded_img = get_response_image(output_path)
        # callback_img = send_file(output_path, mimetype='image/jpeg')
        return {'Status': 'Success', 'Songs': df1.to_json(orient='records')[1:-1].replace('},{', '} {'),
                'ImageBytes': encoded_img}, 200

    except:
        traceback.print_exc()
        return {'message': 'input error'}, 400

    finally:
        pass
        clean_all([
            input_path,
            output_path
        ])


if __name__ == '__main__':
    global upload_directory
    global results_img_directory
    global image_colorizer
    global ALLOWED_EXTENSIONS
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

    upload_directory = './upload/'
    create_directory(upload_directory)

    results_img_directory = './result_images/'
    create_directory(results_img_directory)

    model_directory = './models/'
    create_directory(model_directory)

    port = 5003
    host = "0.0.0.0"

    app.debug = True
    app.run(host=host, port=port, threaded=False)

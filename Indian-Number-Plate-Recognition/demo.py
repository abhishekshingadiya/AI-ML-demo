# Required libraries
import os
import traceback
import io
from base64 import encodebytes

import cv2
import keras_ocr
import numpy as np
import tensorflow as tf
from flask import Flask,request
from flask import send_file
from PIL import Image
import Core.utils as utils
from Core.config import cfg
from Core.yolov4 import YOLOv4, decode
from app_utils import clean_all
from app_utils import download, create_directory
from app_utils import generate_random_filename

app = Flask(__name__)


# gpus = tf.config.experimental.list_physical_devices('CPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]) # limits gpu memory usage
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


pipeline = keras_ocr.pipeline.Pipeline()  # downloads pretrained weights for text detector and recognizer

tf.keras.backend.clear_session()

STRIDES = np.array(cfg.YOLO.STRIDES)
ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, False)
NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
XYSCALE = cfg.YOLO.XYSCALE

size = 608

def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r')  # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG')  # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')  # encode as base64
    return encoded_img


def platePattern(string):
    '''Returns true if passed string follows
    the pattern of indian license plates,
    returns false otherwise.
    '''
    if len(string) < 9 or len(string) > 10:
        return False
    elif string[:2].isalpha() == False:
        return False
    elif string[2].isnumeric() == False:
        return False
    elif string[-4:].isnumeric() == False:
        return False
    elif string[-6:-4].isalpha() == False:
        return False
    else:
        return True


def drawText(img, plates):
    '''Draws recognized plate numbers on the
    top-left side of frame
    '''
    string = 'plates detected :- ' + plates[0]
    for i in range(1, len(plates)):
        string = string + ', ' + plates[i]

    font_scale = 2
    font = cv2.FONT_HERSHEY_PLAIN

    (text_width, text_height) = cv2.getTextSize(string, font, fontScale=font_scale, thickness=1)[0]
    box_coords = ((1, 30), (10 + text_width, 20 - text_height))

    cv2.rectangle(img, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
    cv2.putText(img, string, (5, 25), font, fontScale=font_scale, color=(0, 0, 0), thickness=2)


def plateDetect(frame, input_size, model):
    '''Preprocesses image and pass it to
    trained model for license plate detection.
    Returns bounding box coordinates.
    '''
    frame_size = frame.shape[:2]
    image_data = utils.image_preprocess(np.copy(frame), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    pred_bbox = model.predict(image_data)
    pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)

    bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.25)
    bboxes = utils.nms(bboxes, 0.213, method='nms')

    return bboxes


input_layer = tf.keras.layers.Input([size, size, 3])
feature_maps = YOLOv4(input_layer, NUM_CLASS)
bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, NUM_CLASS, i)
    bbox_tensors.append(bbox_tensor)

model = tf.keras.Model(input_layer, bbox_tensors)
utils.load_weights(model, 'data/YOLOv4-obj_1000.weights')


@app.route("/process", methods=["POST"])
def process_image():
    input_path = generate_random_filename(upload_directory, "jpeg")
    output_path = os.path.join(results_img_directory, os.path.basename(input_path))

    try:
        result = None
        plates = None
        try:
            if 'file' in request.files:
                file = request.files['file']
                if allowed_file(file.filename):
                    file.save(input_path)

            else:
                url = request.json["url"]
                download(url, input_path)

            img = cv2.imread(input_path)  # Reading input
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bboxes = plateDetect(img, size, model)  # License plate detection
            plates = []

            for i in range(len(bboxes)):
                plate_img = img[int(bboxes[i][1]):int(bboxes[i][3]), int(bboxes[i][0]):int(bboxes[i][2])]
                prediction_groups = pipeline.recognize([plate_img])  # Text detection and recognition on license plate
                string = ''
                for j in range(len(prediction_groups[0])):
                    string = string + prediction_groups[0][j][0].upper()

                if platePattern(string) == True and string not in plates:
                    plates.append(string)

            if len(plates) > 0:
                drawText(img, plates)

            img = utils.draw_bbox(img, bboxes)  # Draws bounding box around license plate
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            result = img

            # cv2.imshow("result", img)
            # cv2.waitKey(0)
        finally:
            if result is not None:
                cv2.imwrite(output_path, img)  # Saving output
                print('Output saved to ', output_path)

        encoded_img = get_response_image(output_path)
        # callback_img = send_file(output_path, mimetype='image/jpeg')
        return {'Status': 'Success', 'Number_plat': plates, 'ImageBytes': encoded_img}, 200

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
    try:
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
        port = 5005
        host = "0.0.0.0"

        app.debug = True
        app.run(host=host, port=port, threaded=False)
    except SystemExit:
        pass

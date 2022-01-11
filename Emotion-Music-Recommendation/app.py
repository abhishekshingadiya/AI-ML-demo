import os
import time
from importlib import import_module

# import camera driver
if os.environ.get('CAMERA'):
    # through system
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    # through mimic images
    from camera import Camera
    # from camera_opencv import Camera
    # through opencv
    pass
from flask import Flask, render_template, Response
from camera import *

app = Flask(__name__)

headings = ("Name", "Album", "Artist")
df1 = music_rec()
df1 = df1.head(15)


@app.route('/')
def index():
    print(df1.to_json(orient='records'))
    return render_template('index.html', headings=headings, data=df1)


def gen(camera):
    while True:
        global df1
        frame,df1 = camera.get_frame()
        time.sleep(1)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/t')
def gen_table():
    return df1.to_json(orient='records')


if __name__ == '__main__':
    app.debug = True
    app.run(port=5002)

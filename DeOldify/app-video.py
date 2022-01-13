# import the necessary packages
import traceback
from pathlib import Path

from flask import Flask
from flask import request
from flask import send_file

from app_utils import clean_all
from app_utils import create_directory
from app_utils import download
from app_utils import generate_random_filename
from app_utils import get_model_bin
from deoldify.visualize import *

# Handle switch between GPU and CPU
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    del os.environ["CUDA_VISIBLE_DEVICES"]

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# define a predict function as an endpoint
@app.route("/process", methods=["POST"])
def process_video():
    input_path = generate_random_filename(upload_directory, "mp4")
    output_path = os.path.join(results_video_directory, os.path.basename(input_path))

    try:
        if 'file' in request.files:
            file = request.files['file']
            if allowed_file(file.filename):
                file.save(input_path)
            try:
                render_factor = request.form.getlist('render_factor')[0]
            except:
                render_factor = 30
            output_path = video_colorizer._colorize_from_path(
                Path(input_path), render_factor=render_factor
            )
        else:
            url = request.json["url"]
            download(url, input_path)

            try:
                render_factor = request.json["render_factor"]
            except:
                render_factor = 30

            output_path = video_colorizer.colorize_from_url(
                source_url=url, file_name=input_path, render_factor=render_factor
            )

        callback = send_file(output_path, mimetype="application/octet-stream")

        return callback, 200

    except:
        traceback.print_exc()
        return {"message": "input error"}, 400

    finally:
        clean_all([input_path, output_path])


if __name__ == '__main__':
    global upload_directory
    global results_video_directory
    global video_colorizer
    global ALLOWED_EXTENSIONS
    ALLOWED_EXTENSIONS = set(['mp4'])

    upload_directory = "./upload/"
    create_directory(upload_directory)

    results_video_directory = "./video/result/"
    create_directory(results_video_directory)

    model_directory = "./models/"
    create_directory(model_directory)

    video_model_url = (
        "https://data.deepai.org/deoldify/ColorizeVideo_gen.pth"
    )

    get_model_bin(
        video_model_url, os.path.join(model_directory, "ColorizeVideo_gen.pth")
    )

    video_colorizer = get_video_colorizer()
    video_colorizer.result_folder = Path(results_video_directory)

    port = 5002
    host = "0.0.0.0"

    app.run(host=host, port=port, threaded=False)

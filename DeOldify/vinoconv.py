import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from fastseg import MobileV3Large
from IPython.display import Markdown, display
from openvino.inference_engine import IECore

sys.path.append("../utils")
from notebook_utils import (
    CityScapesSegmentation,
    segmentation_map_to_image,
    viz_result_image,
)

IMAGE_WIDTH = 1024  # Suggested values: 2048, 1024 or 512. The minimum width is 512.
# Set IMAGE_HEIGHT manually for custom input sizes. Minimum height is 512
IMAGE_HEIGHT = 1024 if IMAGE_WIDTH == 2048 else 512
DIRECTORY_NAME = "model"
BASE_MODEL_NAME = DIRECTORY_NAME + f"/fastseg{IMAGE_WIDTH}"

# Paths where PyTorch, ONNX and OpenVINO IR models will be stored
model_path = Path(BASE_MODEL_NAME).with_suffix(".pth")
onnx_path = model_path.with_suffix(".onnx")
ir_path = model_path.with_suffix(".xml")
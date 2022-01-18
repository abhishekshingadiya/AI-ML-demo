import os

import requests
url = 'http://192.168.75.13:5001/process'
file_loc = 'media/dia mirza.jpg'
files = {'file': open(file_loc, 'rb')}
Picture_request = requests.post(url, files=files)
if Picture_request.status_code == 200:
    with open("haar cascade files/images/image.jpg", 'wb') as f:
        f.write(Picture_request.content)

    # os.remove(file_loc)
from io import BytesIO

from PIL import Image

# url = 'http://127.0.0.1:8000/media/2022-01-06-142706_R5XzcHo.jpg'
# urllib.request.urlretrieve(url, "2022-01-06-142706_R5XzcHo.jpg")
# img = Image.open(BytesIO(url.))
# img = Image.open("2022-01-06-142706_R5XzcHo.jpg")
# img.show()

# from PIL import Image
# import urllib.request

# URL = 'http://www.w3schools.com/css/trolltunga.jpg'

# with urllib.request.urlopen(url) as url1:
#     with open('temp.jpg', 'wb') as f:
#         f.write(url1.read())
#
# img = Image.open('temp.jpg')
#
# img.show()

# import requests
#
# res1 = requests.post("http://127.0.0.1:9457/predict",
#                      data={'item_weight': 1, 'item_fat_content': 0,
#                            'item_visibility': 1,
#                            'item_type': 1, 'item_mrp': 123,
#                            'outlet_establishment_year': 2020,
#                            'outlet_size': 2, 'outlet_location_type': 0,
#                            'outlet_type': 0}).text
#
# print(res1)
# from datetime import datetime
#
# a = datetime.today().isoformat()
# print(a)

#
# people = {'Name': {'0': 'INDUSTRY BABY (feat. Jack Harlow)', '1': 'UP', '2': 'Woman', '3': 'Shivers', '4': 'My Universe',
#               '5': 'Leave The Door Open', '6': 'Kiss Me More (feat. SZA)', '7': "Don't Be Shy", '8': 'Butter',
#               '9': 'Love Again', '10': 'MONEY', '11': 'The Motto', '12': "Don't Go Yet",
#               '13': 'Sweet Dreams - Radio Killer Remix', '14': 'Permission to Dance'},
#      'Album': {'0': 'INDUSTRY BABY (feat. Jack Harlow)', '1': 'UP', '2': 'Planet Her', '3': 'Shivers',
#                '4': 'My Universe', '5': 'Leave The Door Open', '6': 'Kiss Me More (feat. SZA)', '7': "Don't Be Shy",
#                '8': 'Butter', '9': 'Future Nostalgia', '10': 'LALISA', '11': 'The Motto', '12': "Don't Go Yet",
#                '13': 'Sweet Dreams (Radio Killer Remix)', '14': 'Butter / Permission to Dance'},
#      'Artist': {'0': 'Lil Nas X', '1': 'INNA', '2': 'Doja Cat', '3': 'Ed Sheeran', '4': 'Coldplay', '5': 'Bruno Mars',
#                 '6': 'Doja Cat', '7': 'Tiësto', '8': 'BTS', '9': 'Dua Lipa', '10': 'LISA', '11': 'Tiësto',
#                 '12': 'Camila Cabello', '13': 'Andra', '14': 'BTS'}}
#
# for p_id, p_info in people.items():
#     print("\nPerson ID:", p_id)
#
#     for key, u in p_info.items():
#         print(key + ':', u)


# a = ['Arbaaz']
#
# b = "".join(a)
# print(b)


# import secrets
#
# # def custom_id():
# #     return secrets.token_urlsafe(8)
#
# print(secrets.token_urlsafe(8))


# img_link = 'https://some-image-link.com/image.png'

# import requests
# from PIL import Image
# import io
#
# response = requests.get(img_link)
# in_memory_file = io.BytesIO(response.content)
# im = Image.open(in_memory_file)
# im.show()

import base64
import io
import PIL.Image as Image
import requests

url = 'http://192.168.75.13:5005/process'
file_loc = 'media/2.jpg'
files = {'file': open(file_loc, 'rb')}
Picture_request = requests.post(url, files=files)
output_data = Picture_request.json()
# number_plat = output_data["Number_plat"]
image_bytes = output_data["ImageBytes"]

b = base64.b64decode(image_bytes)

img = Image.open(io.BytesIO(b))

img.save("Arbaz.jpg")

from datetime import datetime

from django.db import models
from .utils import custom_id

# def upload_path(instance, filename):
#     # change the filename here is required
#     return
#
#
# class ImageInput(models.Model):
#     file_name = models.CharField(max_length=100)
#     input = models.ImageField(upload_to='uploads', default="", null=True)


class History(models.Model):
    # custom_id = models.CharField(primary_key=True, max_length=11, unique=True, default=custom_id)

    his_id = models.BigAutoField(primary_key=True)
    name = models.CharField(max_length=100)
    image_input = models.ImageField(upload_to='media/images_as_input', default="", null=True)
    image_output = models.ImageField(upload_to='media/images_as_output', default="", null=True, blank=True)
    date = models.DateTimeField(default=datetime.now(), blank=True)


class NumberPlateHistory(models.Model):
    num_his_id = models.BigAutoField(primary_key=True)
    num_name = models.CharField(max_length=100)
    num_image_input = models.ImageField(upload_to='media/image_input_number_plate', default="", null=True)
    num_image_output = models.ImageField(upload_to='media/image_output_number_plate', default="", null=True, blank=True)
    num_date = models.DateTimeField(default=datetime.now(), blank=True)

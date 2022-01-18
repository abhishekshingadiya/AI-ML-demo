import base64
import mimetypes
import os
from datetime import datetime
import base64
import io
import PIL.Image as Image
import requests
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from django.http.response import StreamingHttpResponse
from django.shortcuts import render, redirect

from .camera import VideoCamera
from .forms import TourismForm
from .ml_model import logic_layer
from .models import History, NumberPlateHistory

# Create your views here.
res = None


def index(request):
    return render(request=request,
                  template_name='main/index1.html')


def predict(request):
    return render(request=request,
                  template_name='main/predict.html', context={"tourist": res})


def photo_coloring(request):
    try:
        his_prods = []
        for i in History.objects.all():
            his_prods.append(i)
        if len(his_prods) > 14:
            ten = his_prods[-15:]
            return render(request=request, template_name='main/photo_coloring.html',
                          context={'his_prod': ten})
        return render(request=request, template_name='main/photo_coloring.html',
                      context={'his_prod': his_prods})
    except:
        messages.info(request, "Recent Post Not Found")
        return render(request=request, template_name='main/photo_coloring.html')


def detail(request, his_id):
    try:
        product = History.objects.filter(his_id=his_id).first()
        his_prods = []
        for i in History.objects.all():
            his_prods.append(i)
        if len(his_prods) > 9:
            ten = his_prods[-10:]
            context = {'product': product, 'his_prod': ten}

            return render(request, "main/detail.html", context)
        context = {'product': product, 'his_prod': his_prods}
        return render(request, "main/detail.html", context)

    except:
        messages.info(request, "Details Page data not found")
        context = {'product': product, 'his_prod': his_prods}
        return render(request, "main/detail.html", context)


def upload(request):
    try:
        if request.method == 'POST':
            myfile = request.FILES.get("myfile")
            mimetypes.init()

            mimestart = mimetypes.guess_type(myfile.name)[0]

            if mimestart != None:
                mimestart = mimestart.split('/')[0]

                if mimestart in ['video', 'image']:
                    timestr = datetime.today().isoformat()
                    fs = FileSystemStorage(location='media/image_as_input/' + timestr)
                    filename = fs.save(myfile.name, myfile)
                    # uploaded_file_url = fs.url(filename)

                    file_input = os.path.join("media/image_as_input/" + timestr, filename)
                    # file_output = os.path.join("media/image_as_output/", filename)

                    url = 'http://192.168.75.13:5001/process'
                    files = {'file': open(file_input, 'rb')}

                    try:
                        Picture_request = requests.post(url, files=files)
                        if Picture_request.status_code == 200:
                            # Picture_request = requests.post(url, files=files)
                            import base64
                            file = open("media/image_as_output/" + timestr + filename, "wb")
                            file.write(Picture_request.content)
                            file.close()
                            file_loc1 = "media/image_as_output/" + timestr + filename

                            history = History.objects.create(name=filename, image_input=file_input,
                                                             image_output=file_loc1)
                            history.save()
                            with open(file_loc1, "rb") as img_file:
                                image_data = base64.b64encode(img_file.read()).decode('utf-8')
                            uploaded_file = dict()
                            uploaded_file["image"] = image_data
                            uploaded_file = uploaded_file["image"]

                            return render(request=request, template_name='main/upload.html',
                                          context={'uploaded_file': uploaded_file
                                                   })
                        else:
                            messages.info(request, "API is not Connected")
                            return render(request=request,
                                          template_name='main/upload.html')
                    except:
                        messages.info(request, "API is not Connected")
                        return render(request=request,
                                      template_name='main/upload.html')
                else:
                    messages.info(request, "please upload video and image data")
                    return render(request=request, template_name='main/upload.html',
                                  )
            else:
                messages.info(request, "Invalid data")
                return render(request=request, template_name='main/upload.html',
                              )

        else:
            return render(request=request,
                          template_name='main/upload.html')
    except:
        messages.info(request, " Please Check upload data ")
        return render(request=request,
                      template_name='main/upload.html')


def number_plate(request):
    try:
        num_his_prods = []
        for i in NumberPlateHistory.objects.all():
            num_his_prods.append(i)
        return render(request=request, template_name='main/number_plate.html',
                      context={'num_his_prods': num_his_prods})

    except:
        messages.info(request, "Recent Post data is not found")


def detail_for_number_plate(request, his_id):
    try:
        product_for_number_plate = NumberPlateHistory.objects.filter(num_his_id=his_id).first()
        number_his_prods = []
        for i in NumberPlateHistory.objects.all():
            number_his_prods.append(i)
        if len(number_his_prods) > 9:
            ten = number_his_prods[-10:]
            context = {'product_for_number_plate': product_for_number_plate, 'number_his_prods': ten}
            return render(request, "main/detail_for_number_plate.html", context)
        context = {'product_for_number_plate': product_for_number_plate, 'number_his_prods': number_his_prods}
        return render(request, "main/detail_for_number_plate.html", context)
    except:
        messages.info(request, "number detail page not found")
        return render(request, "main/detail_for_number_plate")


def upload_number_plate(request):
    try:
        if request.method == 'POST':
            myfile = request.FILES.get("myfile")
            mimetypes.init()
            mimestart = mimetypes.guess_type(myfile.name)[0]
            if mimestart != None:
                mimestart = mimestart.split('/')[0]

                if mimestart in ['video', 'image']:
                    timestr = datetime.today().isoformat()
                    fs = FileSystemStorage(location='media/image_input_number_plate/' + timestr)
                    filename = fs.save(myfile.name, myfile)
                    # uploaded_file_url = fs.url(filename)

                    file_input = os.path.join("media/image_input_number_plate/" + timestr, filename)
                    # file_output = os.pa``````````````````      h7th.join("media/image_as_output/", filename)

                    url = 'http://192.168.75.13:5005/process'
                    files = {'file': open(file_input, 'rb')}
                    try:
                        Picture_request = requests.post(url, files=files)
                        if Picture_request.status_code == 200:
                            output_data = Picture_request.json()
                            number_plat = output_data["Number_plat"]
                            image_bytes = output_data["ImageBytes"]
                            number_plat1 = "".join(number_plat)
                            import base64
                            file = open("media/image_output_number_plate/" + timestr + filename, "wb")
                            b = base64.b64decode(image_bytes)
                            img = Image.open(io.BytesIO(b))


                            # file.write(image_bytes)
                            # file.close()
                            file_loc1 = "media/image_output_number_plate/" + timestr + filename
                            img.save(file_loc1)

                            number_plate = NumberPlateHistory.objects.create(num_name=filename,
                                                                             num_image_input=file_input,
                                                                             num_image_output=file_loc1)
                            number_plate.save()
                            with open(file_loc1, "rb") as img_file:
                                image_data = base64.b64encode(img_file.read()).decode('utf-8')
                            uploaded_file = dict()
                            uploaded_file["image"] = image_data
                            uploaded_file = uploaded_file["image"]

                            return render(request=request, template_name='main/upload_number_plate.html',
                                          context={'uploaded_file': uploaded_file, 'number_plat': number_plat1})
                        else:
                            messages.info(request, "API Response Not valid")
                            return render(request=request,
                                          template_name='main/upload_number_plate.html')

                    except:
                        messages.info(request, "API is Not Connected")
                        return render(request=request,
                                      template_name='main/upload_number_plate.html')
                else:
                    messages.info(request, "Please Upload Image and Video Data")
                    return render(request=request,
                                  template_name='main/upload_number_plate.html')

            else:
                messages.info(request, "Invalid data")
                return render(request=request,
                              template_name='main/upload_number_plate.html')
        else:
            messages.info(request, 'Number Plate Recognition started')
            return render(request=request,
                          template_name='main/upload_number_plate.html')
    except:
        messages.info(request, "Please Check Upload data")
        return render(request=request,
                      template_name='main/upload_number_plate.html')


def gen(camera):
    while True:
        frame = camera.get_frame1()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def emotion_based_music(request):
    # his_prods = []
    # for i in History.objects.all():
    #     his_prods.append(i)
    # if len(his_prods) > 9:
    #     ten = his_prods[-10:]
    #     return render(request=request, template_name='main/emotion_based_music.html',
    #                   context={'his_prod': ten})
    return render(request=request, template_name='main/emotion_based_music.html')


def upload_image_for_music(request):
    try:
        if request.method == 'POST':
            myfile = request.FILES.get("myfile")
            timestr = datetime.today().isoformat()
            fs = FileSystemStorage(location='media/image_input_for_music/' + timestr)
            filename = fs.save(myfile.name, myfile)
            # uploaded_file_url = fs.url(filename)

            file_input = os.path.join("media/image_input_for_music/" + timestr, filename)
            # file_output = os.pa``````````````````      h7th.join("media/image_as_output/", filename)

            url = 'http://192.168.75.13:5003/process'
            files = {'file': open(file_input, 'rb')}
            Picture_request = requests.post(url, files=files)
            if Picture_request.status_code == 200:
                music_out = Picture_request.json()
                image_byte_data = music_out["ImageBytes"]
                recommended_songs = music_out["Songs"]
                # y = json.loads(recommended_songs)

                import PIL.Image as Image
                import io
                b = base64.b64decode(image_byte_data)
                img = Image.open(io.BytesIO(b))
                img.save(open("media/image_output_for_music/" + timestr + filename, "wb"))

                # file = open("media/image_output_for_music/" + timestr + filename, "wb")
                # file.write(Picture_request.content)
                # file.close()
                file_loc1 = "media/image_output_for_music/" + timestr + filename

                # number_plate = NumberPlateHistory.objects.create(num_name=filename, num_image_input=file_input,
                #                                                  num_image_output=file_loc1)
                # number_plate.save()
                with open(file_loc1, "rb") as img_file:
                    image_data = base64.b64encode(img_file.read()).decode('utf-8')
                uploaded_file = dict()
                uploaded_file["image"] = image_data
                uploaded_file = uploaded_file["image"]

            else:
                messages.info(request, "API Not Connected")
                return render(request=request,
                              template_name='main/music_image_upload.html')

            return render(request=request, template_name='main/music_image_upload.html',
                          context={'uploaded_file': uploaded_file, 'recommended_songs': recommended_songs
                                   })

        else:
            return render(request=request,
                          template_name='main/music_image_upload.html')

    except:
        messages.info(request, "Please Check Upload data")
        return render(request=request,
                      template_name='main/music_image_upload.html')


def driver_drowsiness(request):
    return render(request=request,
                  template_name='main/driver_drowsiness.html')


def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def bigsale(request):
    if request.method == 'POST':
        item_weight = float(request.POST['item_weight'])
        item_fat_content = float(request.POST['item_fat_content'])
        item_visibility = float(request.POST['item_visibility'])
        item_type = float(request.POST['item_type'])
        item_mrp = float(request.POST['item_mrp'])
        outlet_establishment_year = float(request.POST['outlet_establishment_year'])
        outlet_size = float(request.POST['outlet_size'])
        outlet_location_type = float(request.POST['outlet_location_type'])
        outlet_type = float(request.POST['outlet_type'])

        import requests
        res1 = requests.post("http://127.0.0.1:9457/predict",
                             data={'item_weight': item_weight, 'item_fat_content': item_fat_content,
                                   'item_visibility': item_visibility,
                                   'item_type': item_type, 'item_mrp': item_mrp,
                                   'outlet_establishment_year': outlet_establishment_year,
                                   'outlet_size': outlet_size, 'outlet_location_type': outlet_location_type,
                                   'outlet_type': outlet_type})

        # print(res1)

        if res1.status_code == 200:
            res_sale1 = dict()
            res_sale1["pred"] = res1.text
            res_sale = res_sale1["pred"]

            # return render(request=request, template_name='main/photo_coloring.html',
            #               context={'uploaded_file': uploaded_file})
            return render(request=request, template_name='main/bigsale.html', context={'res_sale': res_sale})

        return render(request=request, template_name='main/bigsale.html')

    #         x = [quarter, mode, purpose, year, duration, country, spends, 0.38]
    #         global res
    #         res = logic_layer(x)
    #         return redirect("/predict")
    #     else:
    #         problem = form.errors.as_data()
    #         # This section is used to handle invalid data
    #         messages.error(request, list(list(problem.values())[0][0])[0])
    #         form = TourismForm()
    # form = TourismForm()
    # return render(request=request, template_name='main/index2.html', context={"form": form})
    return render(request=request, template_name='main/bigsale.html')


def index2(request):
    if request.method == 'POST':
        form = TourismForm(request.POST)

        if form.is_valid():

            year = form.cleaned_data['year']
            duration = form.cleaned_data['duration']
            spends = form.cleaned_data['spends'] / 1000
            mode = int(form.cleaned_data['mode'])
            purpose = int(form.cleaned_data['purpose'])
            quarter = int(form.cleaned_data['quarter'])
            country = int(form.cleaned_data['country'])

            x = [quarter, mode, purpose, year, duration, country, spends, 0.38]
            global res
            res = logic_layer(x)
            return redirect("/predict")
        else:
            problem = form.errors.as_data()
            # This section is used to handle invalid data 
            messages.error(request, list(list(problem.values())[0][0])[0])
            form = TourismForm()
    form = TourismForm()
    return render(request=request, template_name='main/index2.html', context={"form": form})


def about(request):
    return render(request=request,
                  template_name="main/about.html")


def under_construction(request):
    messages.info(request, "This page coming soon..")
    return render(request=request,
                  template_name="main/under_construction.html")

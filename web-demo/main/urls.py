from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from main import views

app_name = "main"

urlpatterns = [
    path('', views.index, name="index"),
    path('sales', views.index2, name="index2"),
    path("<int:his_id>", views.detail, name="detail"),
    # path("<int:his_id>", views.detail_for_number_plate, name="detail_for_number_plate"),
    path('bigsales', views.bigsale, name="big"),
    path('upload', views.upload, name='upload'),
    path('upload_number_plate', views.upload_number_plate, name='upload_number_plate'),
    path('upload_image_for_music', views.upload_image_for_music, name='upload_number_plate'),
    path('photo_coloring', views.photo_coloring, name="Coloring"),
    path('emotion_based_music', views.emotion_based_music, name="emotion_based_music"),
    path('number_plate', views.number_plate, name="number_plate"),
    path('driver_drowsiness', views.driver_drowsiness, name="driver_drowsiness"),
    path('video_feed', views.video_feed, name='video_feed'),
    path('predict', views.predict, name="predict"),
    path('about/', views.about, name="about"),
    path('under_construction', views.under_construction, name="under_construction"),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

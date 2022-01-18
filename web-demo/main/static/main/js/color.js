function gphoto(input, ImagePreview) {
        var files = input.files;
        var filesArr = Array.prototype.slice.call(files);

        filesArr.forEach(function(f) {
            if (!f.type.match("image.*")) {
                return;
            }
            var reader = new FileReader();
            reader.onload = function(e) {
                $('.preview').css('margin-top', '4px');
                $($.parseHTML('<img class="gphoto">')).attr('src', e.target.result).appendTo(ImagePreview);
                $($.parseHTML('<input type="hidden" name="photos[]">')).attr('value', e.target.result).appendTo($('.store'));
            };
            reader.readAsDataURL(f);
        });
    }

    $('#gphotos').change(function() {
        gphoto(this, 'div.preview');
    });


    if (request('photos')) {
            $imageArray['gphotos'] = array();

            foreach (request()->photos as $gphoto) {
                $base64_str = substr($gphoto, strpos($gphoto, ",")+1);
                $gphotoPath = 'gphotos/'.\Str::random(11) . '.jpg';
                $gphoto = base64_decode($base64_str);
                \Storage::disk('public')->put($gphotoPath, $gphoto);

               $imageArray['gphotos'][] =  $gphotoPath;
            }

            $imageArray['gphotos'] = json_encode($imageArray['gphotos']);
        }



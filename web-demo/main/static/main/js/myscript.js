
jQuery(document).ready(function($) {
	$('#pics_carousel').flexslider({
		animation: "slide",
		controlNav: true,
        directionNav: false,
		animationLoop: false,
		slideshow: true,
		itemWidth: 140,
		itemMargin: 10,
		maxItems: 3,
		minItems: 4,
		asNavFor: '#pics_slider'
	});

	$('#pics_slider').flexslider({
		animation: "slide",
		controlNav: false,
        directionNav: true,
		animationLoop: false,
		slideshow: true,
		sync: "#pics_carousel"
	});

	$('#vids_carousel').flexslider({
		animation: "slide",
		controlNav: true,
        directionNav: false,
		animationLoop: false,
		slideshow: true,
		itemWidth: 140,
		itemMargin: 10,
		maxItems: 3,
		minItems: 4,
		asNavFor: '#vids_slider'
	});

	$('#vids_slider').flexslider({
		animation: "slide",
		controlNav: false,
        directionNav: true,
		animationLoop: false,
		slideshow: true,
		sync: "#vids_carousel"
	});


	jwplayer("video_1").setup({
		flashplayer: "player.swf",
		file: "http://clips.vorwaerts-gmbh.de/big_buck_bunny.mp4",
		width: 590,
		height: 400,
		stretching: "fill",
		title: "Lorem ipsum dolor sit amet, consectetuer adipiscing elit",
		image: "https://placeimg.com/610/450/tech"
	});
	});$(function() {
	jwplayer("video_2").setup({
		flashplayer: "player.swf",
		file: "http://as1.asset.aparat.com/aparat-video/a_6c9nq52o0qplompnp6p3r64n60qo94n8q13qr7762169-582b__c2dbf.mp4",
		width: 590,
		height: 400,
		stretching: "fill",
		title: "Lorem ipsum dolor sit amet, consectetuer adipiscing elit",
		image: "https://placeimg.com/610/451/tech"
	});

	$(".flex-direction-nav a, .gallery_content .carousel li").on("click", function() {
		$("#vids_slider ul.slides li:not('flex-active-slide')").each(function(i, obj) {
			var vid = $(this).data("vid");
			jwplayer("video_" + vid).pause(true);
		});
	});
});




function readURL(input) {
  if (input.files && input.files[0]) {

    var reader = new FileReader();

    reader.onload = function(e) {
      $('.image-upload-wrap').hide();

      $('.file-upload-image').attr('src', e.target.result);
      $('.file-upload-content').show();

      $('.image-title').html(input.files[0].name);
    };

    reader.readAsDataURL(input.files[0]);

  } else {
    removeUpload();
  }
}

function removeUpload() {
  $('.file-upload-input').replaceWith($('.file-upload-input').clone());
  $('.file-upload-content').hide();
  $('.image-upload-wrap').show();
}
$('.image-upload-wrap').bind('dragover', function () {
    $('.image-upload-wrap').addClass('image-dropping');
  });
  $('.image-upload-wrap').bind('dragleave', function () {
    $('.image-upload-wrap').removeClass('image-dropping');
});
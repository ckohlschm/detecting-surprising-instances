// just grab a DOM element
var element = document.querySelector('#imagemodel')

// And pass it to panzoom
panzoom(element, {
    bounds: true,
    boundsPadding: 0.1
});

/*$('#check-1').click(function() {
    window.alert('Test');
});*/

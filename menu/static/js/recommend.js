$(document).ready(function() {
    $("#movies").change(function(element) {
        $("#movies option:selected").each(function(key, item) {
            console.log(item);
        });
    });
});
var uploadFile = function (e) {
    var freader = new FileReader();
    var file = this.files[0];
    var sendFile = function (edata) {
	var xhr = new XMLHttpRequest();
	var filedata = edata.target.result;
	var form = new FormData();
	xhr.open("POST","upload");
	form.append("files",filedata);
	xhr.send(filedata);
    };
    freader.onload = sendFile;
    freader.readAsBinaryString(file);
};

var file_input = document.getElementById("file-input");
file_input.addEventListener("change",uploadFile,false);

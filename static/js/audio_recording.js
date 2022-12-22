//webkitURL is deprecated but nevertheless
URL = window.URL || window.webkitURL;

var gumStream; 						//stream from getUserMedia()
var rec; 							//Recorder.js object
var input; 							//MediaStreamAudioSourceNode we'll be recording
//soft-coded stuff
let recordLength_ms = 2000; //variable that controls the length of the recorded sample
let left_img_path = "../static/assets/melspec.png" //path for left img
let mid_img_path = "../static/assets/melspec.png" //path for mid img

// shim for AudioContext when it's not avb.
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext //audio context to help us record
var recordButton = document.getElementById("recordButton");
var left_infographic = document.getElementById("left-figure")
var mid_infographic = document.getElementById("mid-figure")
// var door_open_img = document.getElementById("door_open");
// var door_closed_img = document.getElementById("door_closed");
var message_box = document.getElementById("message")
//add events to those 2 buttons
recordButton.addEventListener("click", startRecording);

//original function that starts the recording 
function startRecording() {
	display_message("Recording in progress...")
	console.log("recordButton clicked");
	/*
		Simple constraints object, for more advanced audio features see
		https://addpipe.com/blog/audio-constraints-getusermedia/
	*/

    var constraints = { audio: true, video:false }
 	/*
    	Disable the record button until we get a success or fail from getUserMedia()
	*/
	recordButton.disabled = true;
	/*
    	We're using the standard promise based getUserMedia()
    	https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
	*/
	navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
		console.log("getUserMedia() success, stream created, initializing Recorder.js ...");
		/*
			create an audio context after getUserMedia is called
			sampleRate might change after getUserMedia is called, like it does on macOS when recording through AirPods
			the sampleRate defaults to the one set in your OS for your playback device
		*/
		audioContext = new AudioContext();
		//update the format
		// document.getElementById("formats").innerHTML="Format: 1 channel pcm @ "+audioContext.sampleRate/1000+"kHz"
		/*  assign to gumStream for later use  */
		gumStream = stream;

		/* use the stream */
		input = audioContext.createMediaStreamSource(stream);

		/*
			Create the Recorder object and configure to record mono sound (1 channel)
			Recording 2 channels  will double the file size
		*/
		rec = new Recorder(input,{numChannels:1})

		//start the recording process
		rec.record()

		console.log("Recording started");
		//wait for 5000 millisecond then stop the recording
		setTimeout(stopRecording, recordLength_ms);
	}).catch(function(err) {
	  	//enable the record button if getUserMedia() fails
    	recordButton.disabled = false;
	});
}


function stopRecording() {
	// console.log("stopButton clicked");
	//tell the recorder to stop the recording
	rec.stop();
	recordButton.disabled = false;

	//stop microphone access
	gumStream.getAudioTracks()[0].stop();

	//create the wav blob and pass it on to createDownloadLink
	rec.exportWAV(createDownloadLink);
}

function createDownloadLink(blob) {

	var url = URL.createObjectURL(blob);
	var au = document.createElement('audio');
	var li = document.createElement('div');
	var link = document.createElement('a');

	var filename = new Date().toISOString();


	au.controls = true;
	au.src = url;

	li.appendChild(au);


	var xhr=new XMLHttpRequest();
	xhr.onload=function(e) {
		if(this.readyState === 4) {
			console.log("Server is on.")
		}};
	var fd=new FormData();
	fd.append("audio_data",blob, filename);
    xhr.open("POST","/",true); //Send post request to server, insert backend here
	xhr.send(fd);
	
	$.ajax({
        type: 'POST',
        url: 'http://127.0.0.1:5000/',
        data: fd,
        contentType: false,
        cache: false,
		async:false,
        processData: false,
        success: function(res) {
			// $('#message').text(res).show()
			display_message(res) //send the response to the message element
			update_figures()
			// alert(res)
        },
    });

	// recordingsList.appendChild(li); //commented, used for testing
}

// function openDoor(){
// 	console.log("opening the door")
// 	door_closed_img.style.display = 'none'
// 	door_open_img.style.display = 'block'
// }

// function closeDoor() {
// 	console.log("closing the door")
// 	door_closed_img.style.display = 'block'
// 	door_open_img.style.display = 'none'
// }

//function that updates the message textbox
const display_message = (txt) => {
	//display message from request into the website
	message_box.innerHTML=txt;
}


//function for updating the upper left figure and the middle figure 
const update_figures = () => {
	left_infographic.src = left_img_path
	mid_infographic.src= mid_img_path
}
//webkitURL is deprecated but nevertheless
URL = window.URL || window.webkitURL;

let gumStream; 						//stream from getUserMedia()
let rec; 							//Recorder.js object
let input; 							//MediaStreamAudioSourceNode we'll be recording
//soft-coded stuff
const recordLength_ms = 2000; //variable that controls the length of the recorded sample

let left_figure_path = "../static/assets/project_dev.png" //path for left img
let mid_figure1_path = "../static/assets/dynamic_plot.png" //path for mid img
let mid_figure2_path = "../static/assets/tree_plot.png" //path for mid img

let left_figure = document.getElementById("left-figure")
let mid_figure1 = document.getElementById("mid-figure1")
let mid_figure2 = document.getElementById("mid-figure2")


// Fshim for AudioContext when it's not avb.
let AudioContext = window.AudioContext || window.webkitAudioContext;
let audioContext //audio context to help us record
let recordButton = document.getElementById("recordButton");
let message_box = document.getElementById("message")
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

    let constraints = { audio: true, video:false }
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
	console.log("Recording stopped");
	//tell the recorder to stop the recording
	rec.stop();
	recordButton.disabled = false;

	//stop microphone access
	gumStream.getAudioTracks()[0].stop();

	//create the wav blob and pass it on to createDownloadLink
	rec.exportWAV(createDownloadLink);
}

function createDownloadLink(blob) {

	let url = URL.createObjectURL(blob);
	let au = document.createElement('audio');
	let li = document.createElement('div');
	let link = document.createElement('a');

	let filename = new Date().toISOString();


	au.controls = true;
	au.src = url;

	li.appendChild(au);


	let xhr=new XMLHttpRequest();
	xhr.onload=function(e) {
		if(this.readyState === 4) {
			console.log("Server is on.")
		}};
	let fd=new FormData();
	fd.append("audio_data",blob, filename);
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
			let image = document.getElementById("image");
 
            image.src ="https://media.geeksforgeeks.org/wp-content/uploads/20210915115837/gfg3.png"
        },
    });

}

//function that updates the message textbox
const display_message = (txt) => {
	//display message from request into the website
	message_box.innerHTML=txt;
}


//function for updating the upper left figure and the middle figure 
const update_figures = () => {
	// // create a new timestamp     
	// let timestamp = new Date().getTime(); 
	// //update the left figure    	
	// left_figure.src = "../static/assets/Feature_visuals.png?t=" + timestamp; 
	update_element(left_figure,left_figure_path)
	update_element(mid_figure1, mid_figure1_path)
	update_element(mid_figure2, mid_figure2_path)

}

//function that takes the element and a url, and updates it 
const update_element = (imgElement, imgURL) => {
	 // create a new timestamp 
	 let timestamp = new Date().getTime();  
	 let queryString = "?t=" + timestamp;    
	 imgElement.src = imgURL + queryString;    
}



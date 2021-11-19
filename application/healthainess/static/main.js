/* This script manages communications with the python backend. */

// --- Helper functions

function httpGet(theUrl) {
  var xmlHttp = new XMLHttpRequest();
  xmlHttp.open("GET", theUrl, false); // warning: false for synchronous request
  xmlHttp.send(null);
  return xmlHttp.responseText;
}

// --- UI modification functions

function UI_setVideoStreamEnabled(enabled) {
  if (enabled) { // display "pause" icon

    buttonPause.innerHTML = '<i class="fas fa-pause"></i>';
    buttonPause.title = 'Pause exercise';

  } else { // display "play" icon

    buttonPause.innerHTML = '<i class="fas fa-play"></i>';
    buttonPause.title = 'Play';

  }
}

function UI_setVoiceCommandEnabled(enabled) {
  if (enabled) { // display "mic" icon

    buttonSpeech.innerHTML = '<i class="fas fa-microphone"></i>';
    buttonSpeech.title = 'Disable voice command';

  } else { // display "mic-disabled" icon

    buttonSpeech.innerHTML = '<i class="fas fa-microphone-slash"></i>';
    buttonSpeech.title = 'Enable voice command';

  }
}

// --- State synchronisation
var enabledVideoStream;
var enabledVoiceCommand;

function fetchStates() {
  enabledVideoStream = parseInt(httpGet("/exercise/var/enabledVideoStream"));
  enabledVoiceCommand = parseInt(httpGet("/exercise/var/enabledVoiceCommand"));
}

function applyStates() {
  UI_setVideoStreamEnabled(enabledVideoStream);
  UI_setVoiceCommandEnabled(enabledVoiceCommand);
}

function sendVar_VideoStreamEnabled() {
  var xhr = new XMLHttpRequest();
  xhr.open("POST", "/exercise/var/enabledVideoStream/set");
  xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
  xhr.send(JSON.stringify({ state: enabledVideoStream }));
}

function sendVar_VoiceCommandEnabled() {
  var xhr = new XMLHttpRequest();
  xhr.open("POST", "/exercise/var/enabledVoiceCommand/set");
  xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
  xhr.send(JSON.stringify({ state: enabledVoiceCommand }));
}

// --- Main execution entry

fetchStates();

// Connect buttons to backend
var buttonPause = document.getElementById("pause");   // toggles video stream
var buttonSpeech = document.getElementById("speech"); // toggles voice command

applyStates();

// Set onclicks for state transitions
buttonPause.onclick = function() { // toggle video stream
  enabledVideoStream = !enabledVideoStream;
  UI_setVideoStreamEnabled(enabledVideoStream);
  sendVar_VideoStreamEnabled();
};

buttonSpeech.onclick = function() { // toggle voice command
  enabledVoiceCommand = !enabledVoiceCommand;
  UI_setVoiceCommandEnabled(enabledVoiceCommand);
  sendVar_VoiceCommandEnabled();
};

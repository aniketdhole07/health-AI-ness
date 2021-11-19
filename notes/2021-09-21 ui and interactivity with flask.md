# video

## video - motion jpeg approach
~~~    process generates jpg images -> 
    serves them in a motion jpeg stream on one url
    the stream is embedded in a website served on second url
~~~
- https://towardsdatascience.com/video-streaming-in-web-browsers-with-opencv-flask-93a38846fe00
- https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/


more elaborated on performance and looks at limitations of approach
- https://blog.miguelgrinberg.com/post/video-streaming-with-flask
- https://blog.miguelgrinberg.com/post/flask-video-streaming-revisited
- https://github.com/miguelgrinberg/flask-video-streaming/blob/master/base_camera.py


## video - svg approach

- partial page refresh via poll?
- this polling could be used also to transfer speech commands, other data?

# general interactivity 

- speech commands 
- ..

## websockets

- https://socket.io/docs/v4/
- https://www.fullstackpython.com/websockets.html
- https://flask-socketio.readthedocs.io/en/latest/
- https://github.com/miguelgrinberg/flask-socketio
- https://www.shanelynn.ie/asynchronous-updates-to-a-webpage-with-flask-and-socket-io/
- https://medium.com/swlh/implement-a-websocket-using-flask-and-socket-io-python-76afa5bbeae1



async bidir communication - chatting between server and client.
in the web page use javavascript (typescript) + maybe some libs 


## turbo flask
- based on sockets
- helps updating part of a flask web page
- https://github.com/miguelgrinberg/turbo-flask
- https://turbo.hotwired.dev/ - https://github.com/hotwired/turbo
- https://blog.miguelgrinberg.com/post/dynamically-update-your-flask-web-pages-using-turbo-flask
 

## ajax / polling / intervall / website fetches data 
- https://iq.opengenus.org/single-page-application-with-flask-ajax/
in the web page use javavascript (typescript) + maybe some libs 

- https://medium.com/geekculture/asynchronously-updating-a-webpage-in-a-standard-html-css-js-frontend-8496a3388c01

- https://towardsdatascience.com/talking-to-python-from-javascript-flask-and-the-fetch-api-e0ef3573c451


## full page refresh / intervall 
handmade js

## partial page refresh / intervall 
handmade js


## discussions
- https://www.py4u.net/discuss/998590
  - ajax
  - comet server -> poll 
  - websocket : flask-socketio

- https://stackoverflow.com/questions/68311710/stream-realtime-video-frames-from-user-client-sideandroid-app-iphone-app-pyth


# projects 

## some proprietary parts + mjpeg + socket
- https://alwaysai.co/blog/build-your-own-video-streaming-server-with-flask-socketio
  - https://github.com/alwaysai/video-streamer/blob/main/cv/app.py
  - https://github.com/alwaysai/video-streamer



# our approach

## options

### spa

- use js / css to hide show parts eg. on speech commands

- use full page refresh/rerendering - initiate vs socket? - not really idea of spa
- polling for speech commands? 

### multiple pages 

- initiate browser navigation for speech vs socket?
- polling for speech commands 
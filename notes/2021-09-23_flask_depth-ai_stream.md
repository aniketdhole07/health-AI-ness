Tested the Depth_AI Demo over flask server.

<a href="https://ibb.co/HgzdJMd"><img src="https://i.ibb.co/PChQSbQ/Screenshot-2021-09-23-21-43-26.png" alt="Screenshot-2021-09-23-21-43-26" border="0"></a>

Edited directly the depthai_demo.py file to start flask server 

[Code](../scripts/depthai_flask_stream.py)

And index.html file to templates folder

```
<html>
  <head>
    <title>Video Streaming Demonstration</title>
  </head>
  <body>
    <h1>Video Streaming Demonstration</h1>
    <img  width="500" height="600" src="{{ url_for('video_feed') }}">    
  </body>
</html>
```

ANd modified the `depthai/depthai_helpers/managers.py` file to send the frame instead of displaying with cv2.imshow() : Line 301

```
    def show_frames(self, callback=lambda *a, **k: None):
        for name, frame in self.frames.items():
            if self.mouse_tracker is not None:
                point = self.mouse_tracker.points.get(name)
                value = self.mouse_tracker.values.get(name)
                if point is not None:
                    cv2.circle(frame, point, 3, (255, 255, 255), -1)
                    cv2.putText(frame, str(value), (point[0] + 5, point[1] + 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, str(value), (point[0] + 5, point[1] + 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            return_frame = callback(frame, name)  # Can be None, can be other frame e.g. after copy()
            print(name)
            return return_frame if return_frame is not None else frame
            #cv2.imshow(name, return_frame if return_frame is not None else frame)
```

And the frame was streaming at `http://0.0.0.0:5000/video_feed`

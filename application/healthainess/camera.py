import cv2
import threading
import cv2
import mediapipe as mp

# variables for writing on camera images
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

class VideoCamera(object):
    """A stream to a local webcam. To be replaced with the actual camera stream."""


    def __init__(self):
        # Open a camera stream

        self.cap = cv2.VideoCapture(0)
        # self.cap = blazePose.processImage()
        self.is_record = False
        self.out = None

    def __del__(self):
        self.cap.release()

    # Analyze each image with blazepose


    def get_frame(self):
        ret, frame = self.cap.read()
        with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:
            while self.cap.isOpened():

                if ret:
                    if not ret:
                        print("Ignoring empty camera frame.")
                        # If loading a video, use 'break' instead of 'continue'.
                        continue

                    # Flip the image horizontally for a later selfie-view display, and convert
                    # the BGR image to RGB.
                    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    image.flags.writeable = False
                    results = holistic.process(image)

                    # Draw landmark annotation on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(
                        image,
                        results.face_landmarks,
                        mp_holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles
                            .get_default_pose_landmarks_style())

                    ret, jpeg = cv2.imencode(".jpg", image)

                    if cv2.waitKey(5) & 0xFF == 27:
                        break
                    return jpeg.tobytes()

                else:
                    return None
        self.cap.release()


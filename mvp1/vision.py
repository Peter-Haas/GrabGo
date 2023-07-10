import threading
import cv2
from imutils.video import VideoStream
from object_detection_client import ObjectDetectionClient
from datetime import datetime
import pytz
from object_tracker import MultiObjectTracker
from settings import CAM1_ID


class CamVision:
    """
    Wrapper for input video camera stream and methods that can be performed on the stream.
    Provides each camera stream with an id and method for object detection.
    Exposes Vision API for object detection, recognition and tracking. Also handles events on
    vision.
    """

    def __init__(
        self, src: int = 0, cam_id=CAM1_ID, resolution=(1280, 720), crop=(0, 720, 320, 960)
    ):
        self.src = src

        self.stream = VideoStream(src=src, resolution=resolution, framerate=60)
        # self.stream = cv2.VideoCapture(src)
        # self.frame = None
        # self.grabbed = False

        self.image_np = None
        self.crop = crop  # [y1:y2, x1:x2]
        self.detected_objects = None
        self.object_tracker = None
        self.cam_id = cam_id

    def vision_log(self, log_message):
        print(f"[INFO] Cam: {self.cam_id}: {log_message}")

    def start(self):
        self.vision_log(f"Starting camera {self.src}...")
        self.stream.start()

    # def read(self):
    #     (self.grabbed, self.frame) = self.stream.read()
    #     return self.frame

    def stop(self):
        # do a bit of cleanup
        self.vision_log("Cleaning up...")
        cv2.destroyAllWindows()
        self.stream.stop()

    def detect_objects(
        self,
        save_to_file: bool = False,
        input_dir_path: str = "",
        convert_to_rgb: bool = False,
        object_tracking: bool = False,
    ):
        """
        Uses the ObjectDetectionClient to detect objects in the image sampled at the call time.
        Returns list of objects' bounding boxes
        :param save_to_file:
        :param input_dir_path:
        :param convert_to_rgb:
        :return:
        """
        self.image_np = self.stream.read()  # self.read()

        crop = self.crop
        if crop:
            self.image_np = self.image_np[crop[0] : crop[1], crop[2] : crop[3]]

        if save_to_file:
            filename = (
                input_dir_path
                + "/sample-"
                + datetime.now().astimezone(pytz.utc).strftime("%Y_%m_%d_%H_%M_%S")
                + ".jpg"
            )
            cv2.imwrite(filename, self.image_np)

        if convert_to_rgb:
            self.image_np = cv2.cvtColor(self.image_np, cv2.COLOR_BGR2RGB)

        _, input_image = cv2.imencode(".jpg", self.image_np)
        input_image = input_image.tostring()

        ## If image from file, do this:
        # input_image = open(filename, "rb").read()

        obc = ObjectDetectionClient(image=input_image)
        # print("Detecting objects...")
        detection_results = obc.inference()
        self.vision_log(f"{len(detection_results)} items detected")
        self.detected_objects = detection_results
        if self.detected_objects and object_tracking:
            self.object_tracker = MultiObjectTracker(
                cam_vision=self, detected_objects=self.detected_objects
            )

    def recognize_objects(self):
        """
        Use cam crops and detected objects and pass then to object recognition pipeline
        """
        pass

    def track_objects(self):
        if self.object_tracker:
            self.object_tracker.track_objects()

    def handle_significant_change_while_tracking(self):
        """
        Make ObjectTracker call this when a significant movement is detected while object tracking
        """
        pass

    def start_continuous_detection_and_tracking(self):
        """
        Implement custom vision based logic here
        """
        self.detect_objects(object_tracking=True)
        while True:
            self.track_objects()

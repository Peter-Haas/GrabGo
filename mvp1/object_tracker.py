import random
import string
import time

import cv2
import dlib

from constants import ITEM_STATE_IDLE, ITEM_STATE_MOVING
from settings import (
    VISUALIZE_VISION_OUTPUT,
    CAM1_DOOR_WALL,
    CAM1_ID,
    CAM2_ID,
    VISION_SAMPLES,
    MOTION_DETECTION_MARGIN,
)


class Tracker:
    """
    Tracks one item using bounding box and label.
    """

    def __init__(
        self,
        track: bool = True,
        visualize: bool = VISUALIZE_VISION_OUTPUT,
        bounding_box: tuple = (0, 0, 0, 0),
        label: str = "",
    ):

        self.track = track
        self.visualize = visualize
        self.bounding_box = bounding_box  # (startX, startY, endX, end)
        self.label = label
        self.status = ITEM_STATE_IDLE
        self.prev_averaged_bounding_box = (0, 0, 0, 0)  # bounding_box

    def start_tracking(self, image_np_rgb):
        # construct a dlib rectangle object from the bounding
        # box coordinates and start the correlation tracker
        t = dlib.correlation_tracker()
        rect = dlib.rectangle(
            self.bounding_box[0], self.bounding_box[1], self.bounding_box[2], self.bounding_box[3]
        )
        t.start_track(image_np_rgb, rect)
        return t

    def draw_bounding_boxes_on(self, image_np):
        # grab the corresponding class label for the detection
        # and draw the bounding box
        cv2.rectangle(
            image_np,
            (self.bounding_box[0], self.bounding_box[1]),
            (self.bounding_box[2], self.bounding_box[3]),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            image_np,
            self.label,
            (self.bounding_box[0], self.bounding_box[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            2,
        )


class MultiObjectTracker:
    """
    Takes input as items detected from ObjectDetectionClient pipeline (as labels and bounding
    boxes) and the corresponding CamVision object and creates Tracker instances for each item.
    Tracks each item by updating their bounding boxes using dlib. Counts items removed or added
    into the boundaries.
    """

    def __init__(self, cam_vision, detected_objects: list, vision_samples: int = VISION_SAMPLES):
        self.cam_vision = cam_vision
        self.crop = self.cam_vision.crop
        self.detected_objects = detected_objects
        # tracker for each detected object
        self.trackers = []
        # labels for each tracked object
        self.labels = []
        self.vision_samples = vision_samples
        self._measureCnt = 0
        self._events = list(range(vision_samples))

    def detect_significant_movement(self, old_bounding_box: tuple, new_bounding_box: tuple) -> bool:
        # box = (startX, startY, endX, end)
        # center = ((startX + (endX - startX) / 2), (startY + (endY - startY) / 2))
        old_center = (
            (old_bounding_box[0] + (old_bounding_box[2] - old_bounding_box[0]) / 2),
            (old_bounding_box[1] + (old_bounding_box[3] - old_bounding_box[1]) / 2),
        )
        new_center = (
            (new_bounding_box[0] + (new_bounding_box[2] - new_bounding_box[0]) / 2),
            (new_bounding_box[1] + (new_bounding_box[3] - new_bounding_box[1]) / 2),
        )

        if (
            new_center[0] > old_center[0] + MOTION_DETECTION_MARGIN
            or new_center[0] < old_center[0] - MOTION_DETECTION_MARGIN
        ) or (
            new_center[1] > old_center[1] + MOTION_DETECTION_MARGIN
            or new_center[1] < old_center[1] - MOTION_DETECTION_MARGIN
        ):
            return True
        return False

    def track_objects(self, convert_to_rgb=True):
        self.image_np = self.cam_vision.stream.read()  # self.cam_vision.read()

        if self.crop:
            self.image_np = self.image_np[self.crop[0] : self.crop[1], self.crop[2] : self.crop[3]]

        image_np_rgb = None
        if convert_to_rgb:
            image_np_rgb = cv2.cvtColor(self.image_np, cv2.COLOR_BGR2RGB)

        if len(self.trackers) == 0:
            for item in self.detected_objects:
                # compute the (x, y)-coordinates of the bounding box
                # for the object
                bounding_box = (item["xmin"], item["ymin"], item["xmax"], item["ymax"])
                label = (
                    item["class"]
                    + "_"
                    + "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
                )

                tracker = Tracker(bounding_box=bounding_box, label=label)
                t = tracker.start_tracking(image_np_rgb=image_np_rgb)
                self.trackers.append((tracker, t))
                tracker.draw_bounding_boxes_on(self.image_np)
        else:
            # loop over each of the trackers
            for tracker, t in self.trackers:
                # update the tracker and grab the position of the tracked object
                # tracker: Tracker, t: dlib.start_track
                if tracker.track:
                    t.update(image_np_rgb)
                    pos = t.get_position()

                    # unpack the position object
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    # update tracker's bounding box
                    tracker.bounding_box = (startX, startY, endX, endY)

                    # First, we average out the new_total_weights over WEIGHT_SAMPLE number of samples
                    self._events[self._measureCnt] = (startX, startY, endX, endY)
                    self._measureCnt += 1
                    if self._measureCnt == self.vision_samples:
                        self._measureCnt = 0
                        self._sum = (0, 0, 0, 0)
                        for x in range(0, self.vision_samples - 1):
                            # self._sum += self._events[x]
                            self._sum = [self._sum[i] + j for i, j in enumerate(self._events[x])]
                        # averaged_bounding_box = self._sum / self.vision_samples
                        averaged_bounding_box = [i / self.vision_samples for i in self._sum]
                        averaged_bounding_box = (
                            averaged_bounding_box[0],
                            averaged_bounding_box[1],
                            averaged_bounding_box[2],
                            averaged_bounding_box[3],
                        )

                        has_moved = False
                        if not tracker.prev_averaged_bounding_box == (0, 0, 0, 0):
                            has_moved = self.detect_significant_movement(
                                tracker.prev_averaged_bounding_box, averaged_bounding_box
                            )

                        if has_moved:
                            if tracker.status == ITEM_STATE_IDLE:
                                self.cam_vision.vision_log(f"{tracker.label} moving")
                                tracker.status = ITEM_STATE_MOVING
                                # confer with weight

                                # if self.continuous_object_detection:
                                #     time.sleep(3.0)
                                #     # stop tracking that object
                                #     tracker.track = False
                                #
                                #     # re-detect objects
                                #     detected_o = self.cam_vision.detect_objects()
                                #     self.detected_objects = detected_o
                                #     self.trackers = []
                                #     break
                        else:
                            if tracker.status == ITEM_STATE_MOVING:
                                self.cam_vision.vision_log(f"{tracker.label} idle")
                                tracker.status = ITEM_STATE_IDLE

                        tracker.prev_averaged_bounding_box = averaged_bounding_box
                        if tracker.visualize:
                            tracker.draw_bounding_boxes_on(self.image_np)

        # show the output frame
        if VISUALIZE_VISION_OUTPUT:
            # print(f"{self.image_np.shape[0]}, {self.image_np.shape[1]}")
            cv2.imshow("Frame", self.image_np)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return

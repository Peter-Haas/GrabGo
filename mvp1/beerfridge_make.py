import os
import time

from wiiboard import Wiiboard
from door import Door
from vision import CamVision
from concurrent.futures import ProcessPoolExecutor

from settings import WEIGHT_SENSORS_ENABLED, VISION_ENABLED, CAM_IDS, CAM_RESOLUTIONS, CAM_CROPS


def wiiBoardProcess():
    print("[INFO] executing weight task on process: {}".format(os.getpid()))
    board = Wiiboard()
    board.initialize()
    board.perform_initial_calibration()
    board.run_weight_event_loop()


def camVisionProcess(src, cam_id):
    print("[INFO] executing vision task on process: {}".format(os.getpid()))
    # Initialize and start cameras
    cam_vision = CamVision(
        src=src, cam_id=cam_id, resolution=CAM_RESOLUTIONS[cam_id], crop=CAM_CROPS[cam_id]
    )
    # Start the cameras and wait 3 seconds for cold boot-up to complete
    cam_vision.start()
    time.sleep(3.0)
    cam_vision.start_continuous_detection_and_tracking()


def main():
    print("[INFO] executing main on process: {}".format(os.getpid()))
    door = Door()
    with ProcessPoolExecutor(max_workers=8) as executor:
        tasks = []
        if VISION_ENABLED:
            for src, cam_id in enumerate(CAM_IDS):
                tasks.append(executor.submit(camVisionProcess, src, cam_id))

        if WEIGHT_SENSORS_ENABLED:
            tasks.append(executor.submit(wiiBoardProcess))

        print(f"*********")


if __name__ == "__main__":
    main()

import configparser
import os
from typing import Tuple

import cv2
import numpy as np
from loguru import logger
from video.video_stream import VideoStream

from .calibration import calibrate_hsv


class BehaviorMatching(object):
    def __init__(self):
        self.HSV_CALIBRATION_KEY = "HSVCalibration"

        self.reconstruction_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reconstruction.ini")
        self.reconstruction_config = configparser.ConfigParser()

        if os.path.exists(self.reconstruction_file):
            try:
                self.reconstruction_config.read_file(open(self.reconstruction_file))
            except Exception as e:
                logger.error(f"Failed to parse config: {e}")

        self.exporting = True
        self.stream = VideoStream()

    @property
    def lower_bound_color(self) -> Tuple[int, int, int]:
        config = self.reconstruction_config[self.HSV_CALIBRATION_KEY]
        return (int(config["h_min"]), int(config["s_min"]), int(config["v_min"]))

    @property
    def upper_bound_color(self) -> Tuple[int, int, int]:
        config = self.reconstruction_config[self.HSV_CALIBRATION_KEY]
        return (int(config["h_max"]), int(config["s_max"]), int(config["v_max"]))

    def calibrate(self):
        config = calibrate_hsv()
        if self.HSV_CALIBRATION_KEY in self.reconstruction_config:
            logger.warning("Overriding existing reconstruction entry")
        self.reconstruction_config[self.HSV_CALIBRATION_KEY] = dict(config._asdict())

        with open(self.reconstruction_file, "w+") as f:
            self.reconstruction_config.write(f)

    def transform_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.HSV_CALIBRATION_KEY not in self.reconstruction_config:
            logger.warning(f"Config Option: {self.HSV_CALIBRATION_KEY} not found, starting calibration.")
            self.calibrate()

        frame = cv2.flip(frame, 5)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.lower_bound_color, self.upper_bound_color)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        frameray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(frameray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            # for cnt in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), [255, 0, 0], 2)
            cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
        return frame

    def start_matching(self):
        self.stream.start(self.transform_frame)

    def stop_matching(self):
        self.stream.stop()

import configparser
import os
from typing import Tuple

import cv2
import numpy as np
from loguru import logger

from femflow.video.video_stream import VideoStream


class BehaviorMatching(object):
    def __init__(self):
        self.HSV_CALIBRATION_KEY = "HSVCalibration"

        self.reconstruction_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "reconstruction.ini"
        )
        self.reconstruction_config = configparser.ConfigParser()

        if os.path.exists(self.reconstruction_file):
            try:
                self.reconstruction_config.read_file(open(self.reconstruction_file))
            except Exception as e:
                logger.error(f"Failed to parse config: {e}")

        # self.stream = VideoStream("enxf2379f07f0e2", resolution=(640, 480))
        self.stream = VideoStream(2)
        self.mask = np.array([])
        self.frame = np.array([])

        # Convergence helpers
        self._last_radius = 0
        self._last_thickness = 0

        self.current_rectangle_height: int = 0

        self.first_frame = True

    @property
    def lower_bound_color(self) -> Tuple[int, int, int]:
        config = self.reconstruction_config[self.HSV_CALIBRATION_KEY]
        return (int(config["h_min"]), int(config["s_min"]), int(config["v_min"]))

    @property
    def upper_bound_color(self) -> Tuple[int, int, int]:
        config = self.reconstruction_config[self.HSV_CALIBRATION_KEY]
        return (int(config["h_max"]), int(config["s_max"]), int(config["v_max"]))

    @property
    def streaming(self):
        return self.stream.streaming

    def destroy(self):
        self.stream.destroy()

    def transform_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.HSV_CALIBRATION_KEY not in self.reconstruction_config:
            raise RuntimeError("No calibration found! Run the calibrator first")

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        self.mask = cv2.inRange(hsv, self.lower_bound_color, self.upper_bound_color)
        self.mask = cv2.erode(self.mask, None, iterations=2)
        self.mask = cv2.dilate(self.mask, None, iterations=2)
        res = cv2.bitwise_and(frame, frame, mask=self.mask)
        mask_three_channel = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)

        frameray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(
            frameray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(contour)
            self.current_rectangle_height = h
            cv2.rectangle(frame, (x, y), (x + w, y + h), [255, 0, 0], 2)

            min_y = np.amin(contour, axis=0)[0][1] * 1.1

            top_contours = np.array([row for row in contour if row[0][1] <= min_y])
            contours = np.array([row for row in contour if row[0][1] > min_y])

            cv2.drawContours(frame, top_contours, -1, (0, 0, 255), 3)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

        return np.hstack((frame, mask_three_channel))

    def start_matching(self):
        self.stream.start(self.transform_frame)

    def stop_matching(self):
        self.stream.stop()

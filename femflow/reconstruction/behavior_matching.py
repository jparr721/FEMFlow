import configparser
import os
from typing import Generator, Tuple

import cv2
import numpy as np
from loguru import logger
from OpenGL.GL import *
from utils.graphics.textures import build_texture, load_texture_from_image

from .calibration import TEXTURES_PATH, calibrate_hsv


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

        self.exporting = False
        self.capture_texture = -1
        self.w = 0
        self.h = 0

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

    def display(self):
        if self.HSV_CALIBRATION_KEY not in self.reconstruction_config:
            logger.warning(f"Config Option: {self.HSV_CALIBRATION_KEY} not found, starting calibration.")
            self.calibrate()
        self.capture_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.capture_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        self.exporting = True
        _, w, h = load_texture_from_image(self.capture_texture, self.export_bb_realtime())
        self.w = w
        self.h = h

    def export_bb_realtime(self) -> Generator[np.ndarray, None, None]:
        cap = cv2.VideoCapture(0)
        img = np.zeros((100, 100))
        while self.exporting:
            _, img = cap.read()
            img = cv2.flip(img, 5)

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, self.lower_bound_color, self.upper_bound_color)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            res = cv2.bitwise_and(img, img, mask=mask)

            imgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(imgray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                # for cnt in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img, (x, y), (x + w, y + h), [255, 0, 0], 2)
                cv2.drawContours(img, contour, -1, (0, 255, 0), 3)

            # Generate the output data stream as a generator to avoid thread annihilation
            data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.exporting = False
            return data

        filename = f"{TEXTURES_PATH}/bounding_box.png"
        logger.info(f"Writing texture {filename}")
        cv2.imwrite(filename, img)

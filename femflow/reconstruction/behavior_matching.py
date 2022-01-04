import configparser
import os
from typing import List, Tuple

import cv2
import numpy as np
from loguru import logger

from femflow.numerics.linear_algebra import distance
from femflow.video.video_stream import VideoStream


class BehaviorMatching(object):
    def __init__(self):
        self.HSV_CALIBRATION_KEY = "HSVCalibration"

        self.reconstruction_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "reconstruction.ini"
        )
        self.reconstruction_config = configparser.ConfigParser()
        self.mask_directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "masks"
        )
        self.texture_directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "textures"
        )

        if os.path.exists(self.reconstruction_file):
            try:
                self.reconstruction_config.read_file(open(self.reconstruction_file))
            except Exception as e:
                logger.error(f"Failed to parse config: {e}")

        self.stream = VideoStream()
        self.mask = np.array([])
        self.frame = np.array([])

        self.radius_convergence_reached = False
        self.thickness_convergence_reached = False

        self._void_radii: List[int] = []
        self._beam_thicknesses: List[int] = []

        # Convergence helpers
        self._last_radius = 0
        self._last_thickness = 0

        self._convergence_patience = 100
        self._last_radius_patience_threshold = 0
        self._last_thickness_patience_threshold = 0

        self.starting_calibrated_rectangle_height: int = 0
        self.ending_calibrated_rectangle_height: int = 0
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

    @property
    def void_radius(self):
        if not self.radius_convergence_reached:
            return float(
                np.average(self._void_radii) if len(self._void_radii) != 0 else 0
            )
        else:
            return float(self._last_radius)

    @property
    def beam_thickness(self):
        if not self.thickness_convergence_reached:
            return float(
                np.average(self._beam_thicknesses)
                if len(self._beam_thicknesses) != 0
                else 0
            )
        else:
            return float(self._last_thickness)

    @property
    def strain_pct(self) -> float:
        if self.starting_calibrated_rectangle_height > 0.0:
            return 100 - (
                (
                    self.ending_calibrated_rectangle_height
                    / self.starting_calibrated_rectangle_height
                )
                * 100
            )
        else:
            return 0.0

    def set_starting_calibrated_rectangle_height(self):
        self.starting_calibrated_rectangle_height = self.current_rectangle_height

    def set_ending_calibrated_rectangle_height(self):
        self.ending_calibrated_rectangle_height = self.current_rectangle_height

    def save_frame(self):
        pass

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
            # for cnt in contours:
            x, y, w, h = cv2.boundingRect(contour)

            self.current_rectangle_height = h

            cv2.rectangle(frame, (x, y), (x + w, y + h), [255, 0, 0], 2)
            cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)

        circles = cv2.HoughCircles(
            self.mask,
            cv2.HOUGH_GRADIENT,
            1,
            1,
            param1=100,
            param2=25,
            minRadius=0,
            maxRadius=100,
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype(np.int32)
            self._save_beam_thickness(circles)
            for x, y, r in circles:
                self._save_radius(r)

                # The circle
                cv2.circle(frame, (x, y), r, (0, 0, 255), 2)

                # The center
                cv2.circle(frame, (x, y), 1, (0, 0, 255), 2)

        if cv2.waitKey(10) & 0xFF == ord("s"):
            logger.info("Saving image mask and source")
            cv2.imwrite(os.path.join(self.mask_directory, "mask.png"), self.mask)
            cv2.imwrite(os.path.join(self.texture_directory, "texture.png"), frame)
        frame = np.hstack((frame, mask_three_channel))
        self._compute_convergences()
        return frame

    def start_matching(self):
        self.stream.start(self.transform_frame)

    def stop_matching(self):
        self.stream.stop()

    def _compute_convergences(self):
        if not self.radius_convergence_reached:
            rad = self.void_radius
            if np.isclose(self._last_radius, rad, atol=0.5):
                self._last_radius_patience_threshold += 1
            self._last_radius = rad
            if self._last_radius_patience_threshold >= self._convergence_patience:
                self.radius_convergence_reached = True
                self._void_radii.clear()  # Free this memory
        if not self.thickness_convergence_reached:
            thicc = self.beam_thickness
            if np.isclose(self._last_thickness, thicc) and thicc > 0.1:
                self._last_thickness_patience_threshold += 1
            self._last_thickness = thicc
            if self._last_thickness_patience_threshold >= self._convergence_patience:
                self.thickness_convergence_reached = True
                self._beam_thicknesses.clear()

    def _save_radius(self, r: int):
        if not self.radius_convergence_reached:
            self._void_radii.append(r)

    def _save_beam_thickness(self, circles: List[np.ndarray]):
        if not self.thickness_convergence_reached:
            if len(circles) > 1:
                for i, (x1, y1, _) in enumerate(circles):
                    for j, (x2, y2, _) in enumerate(circles):
                        if i == j:
                            continue
                        lcircle = np.array([x1, y1])
                        rcircle = np.array([x2, y2])
                        self._beam_thicknesses.append(int(distance(lcircle, rcircle)))


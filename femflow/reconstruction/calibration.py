import os
from collections import namedtuple

import cv2
import numpy as np
from loguru import logger

HSVCalibration = namedtuple("HSVCalibration", ["h_min", "s_min", "v_min", "h_max", "s_max", "v_max"])
MASKS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "masks")
TEXTURES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textures")


def calibrate_hsv() -> HSVCalibration:
    cap = cv2.VideoCapture(0)

    # Create a window
    cv2.namedWindow("Calibration")

    # Create trackbars for color change
    # Hue is from 0-179 for Opencv
    cv2.createTrackbar("h_min", "Calibration", 0, 179, lambda x: x)
    cv2.createTrackbar("s_min", "Calibration", 0, 255, lambda x: x)
    cv2.createTrackbar("v_min", "Calibration", 0, 255, lambda x: x)
    cv2.createTrackbar("h_max", "Calibration", 0, 179, lambda x: x)
    cv2.createTrackbar("s_max", "Calibration", 0, 255, lambda x: x)
    cv2.createTrackbar("v_max", "Calibration", 0, 255, lambda x: x)

    # Set default value for Max HSV trackbars
    cv2.setTrackbarPos("h_max", "Calibration", 179)
    cv2.setTrackbarPos("s_max", "Calibration", 255)
    cv2.setTrackbarPos("v_max", "Calibration", 255)

    # Initialize HSV min/max values
    h_min = s_min = v_min = h_max = s_max = v_max = 0
    ph_min = ps_min = pv_min = ph_max = ps_max = pv_max = 0

    while True:
        _, image = cap.read()
        image = cv2.flip(image, 5)

        # Get current positions of all trackbars
        h_min = cv2.getTrackbarPos("h_min", "Calibration")
        s_min = cv2.getTrackbarPos("s_min", "Calibration")
        v_min = cv2.getTrackbarPos("v_min", "Calibration")
        h_max = cv2.getTrackbarPos("h_max", "Calibration")
        s_max = cv2.getTrackbarPos("s_max", "Calibration")
        v_max = cv2.getTrackbarPos("v_max", "Calibration")

        # Set minimum and maximum HSV values to display
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        # Convert to HSV format and color threshold
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)

        # Print if there is a change in HSV value
        if (
            (ph_min != h_min)
            | (ps_min != s_min)
            | (pv_min != v_min)
            | (ph_max != h_max)
            | (ps_max != s_max)
            | (pv_max != v_max)
        ):
            logger.info("New Values")
            logger.info(f"h_min {h_min} s_min {s_min} v_min {v_min} h_max {h_max} s_max {s_max} v_max {v_max}")
            ph_min = h_min
            ps_min = s_min
            pv_min = v_min
            ph_max = h_max
            ps_max = s_max
            pv_max = v_max

        # Display result image
        cv2.imshow("Calibration", result)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    return HSVCalibration(h_min, s_min, v_min, h_max, s_max, v_max)


def calibrate_mask():
    cap = cv2.VideoCapture(0)
    img = None
    while True:
        _, img = cap.read()
        img = cv2.flip(img, 5)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # These parameters _must_ be these values to accurately capture the square
        lower_color = (40, 70, 115)
        upper_color = (80, 255, 255)

        mask = cv2.inRange(hsv, lower_color, upper_color)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cv2.imshow("Camera", mask)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            cap.release()
            break

    cv2.destroyAllWindows()
    filename = os.path.join(MASKS_PATH, "mask_calibration.png")
    logger.info(f"Saving calibration mask image to {filename}")
    cv2.imwrite(filename, mask)


def calibrate_bb():
    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        img = cv2.flip(img, 5)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # These parameters _must_ be these values to accurately capture the square
        lower_color = (40, 70, 115)
        upper_color = (80, 255, 255)

        mask = cv2.inRange(hsv, lower_color, upper_color)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        res = cv2.bitwise_and(img, img, mask=mask)

        imgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(imgray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            # for cnt in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), [255, 0, 0], 2)
            cv2.drawContours(img, contour, -1, (0, 255, 0), 3)

        cv2.imshow("Camera", img)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            cap.release()
            break

    cv2.destroyAllWindows()

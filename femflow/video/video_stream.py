import threading

import cv2
import numpy as np
from loguru import logger
from typing_extensions import Protocol


class TransformationFn(Protocol):
    def __call__(self, a: np.ndarray) -> np.ndarray:
        ...


class VideoStream(object):
    def __init__(self):
        self.stream = cv2.VideoCapture(0)
        _, self.frame = self.stream.read()
        self.streaming = False
        self.displaying = True

    def destroy(self):
        self.stream.release()

    def start(self, transform: TransformationFn):
        if not self.streaming:
            logger.info("Starting background task")
            self.streaming = True
            self.displaying = True
            threading.Thread(target=self.update, args=(transform,), daemon=True).start()

    def stop(self):
        self.streaming = False
        self.displaying = False

    def update(self, transform: TransformationFn):
        while True:
            if not self.streaming:
                break
            _, self.frame = self.stream.read()
            self.frame = transform(self.frame)
            if self.frame is None:
                raise ValueError("Transform function returned 'None'")

            if self.displaying:
                cv2.imshow("Stream", self.frame)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    self.stop()
                    cv2.destroyAllWindows()

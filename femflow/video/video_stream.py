import threading
from typing import Union

import cv2
import numpy as np
from loguru import logger
from typing_extensions import Protocol


class TransformationFn(Protocol):
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        ...


class VideoStream(object):
    def __init__(self, stream_source="builtin", resolution=(-1, -1)):
        self.resolution = resolution
        self.source_address = 0
        self._register_stream_source(stream_source)

        self.reset()

    def reset(self):
        self.stream = cv2.VideoCapture(self.source_address)
        _, self.frame = self.stream.read()
        self.streaming = False
        self.displaying = False

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
        if "camera" in self.__dict__:
            self.camera.stopWebcam()

    def update(self, transform: TransformationFn):
        while True:
            if not self.streaming:
                break
            _, self.frame = self.stream.read()
            if self.frame is not None:

                if self.resolution != (-1, -1):
                    self.frame = cv2.resize(self.frame, self.resolution)

                self.frame = transform(self.frame)
                if self.frame is None:
                    raise ValueError("Transform function returned 'None'")

                if self.displaying:
                    cv2.imshow("Stream", self.frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.stop()
                        cv2.destroyAllWindows()
                else:
                    self.stop()
                    cv2.destroyAllWindows()
        logger.success("Thread exited successfully")

    def _register_gopro(self, stream_source: str):
        from goprocam import GoProCamera

        addr = GoProCamera.GoPro.getWebcamIP(stream_source)
        self.camera = GoProCamera.GoPro(ip_address=addr)
        self.camera.startWebcam()
        self.source_address = f"udp://{addr}:8554"

    def _register_stream_source(self, stream_source: Union[str, int]):
        if stream_source == "builtin":
            self.source_address = 0
        elif isinstance(stream_source, int):
            self.source_address = stream_source
        else:
            self._register_gopro(stream_source)

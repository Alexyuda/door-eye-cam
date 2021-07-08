import cv2
import time

from datetime import datetime
from pipeline.pipeline import Pipeline


class CaptureVideo(Pipeline):
    """Pipeline task to capture video stream from file or webcam."""

    def __init__(self, src=0, select_roi=False):

        self.src = src
        self.cap = cv2.VideoCapture(src)

        if select_roi:
            ret, image = self.cap.read()
            self.roi = cv2.selectROI(image)
            cv2.destroyAllWindows()
        else:
            self.roi = []

        if not self.cap.isOpened():
            raise IOError(f"Cannot open video {src}")

        self.fps = 25
        self.last_frame_time = datetime.now()
        self.total_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(src, str) else -1
        self.frame_count = 0
        super(CaptureVideo, self).__init__()

    def generator(self):
        """Yields the frame content and metadata."""

        frame_idx = 0
        while self.has_next():
            try:
                ret, image = self.cap.read()

                if not ret:
                    # no frames has been grabbed
                    time.sleep(1)
                    self.cap = cv2.VideoCapture(self.src)
                    print(f"Trying to reconnect to {self.cap}")
                    continue

                if self.roi:
                    image = image[int(self.roi[1]):int(self.roi[1] + self.roi[3]),
                                  int(self.roi[0]):int(self.roi[0] + self.roi[2])]

                cur_time = datetime.now()
                self.frame_count += 1
                cur_fps = 1 / (cur_time - self.last_frame_time).total_seconds()
                self.fps = self.fps * (self.frame_count - 1) / self.frame_count + cur_fps / self.frame_count
                print(self.fps)
                self.last_frame_time = cur_time

                data = {
                    "frame_idx": frame_idx,
                    "image_id": f"{frame_idx:06d}",
                    "image": image,
                    "time": datetime.now(),
                    "fps": self.fps
                }

                if self.filter(data):
                    frame_idx += 1
                    yield self.map(data)
            except StopIteration:
                return

    def cleanup(self):
        """Closes video file or capturing device.

        This function should be triggered after the pipeline completes.
        """

        self.cap.release()

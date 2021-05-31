import cv2

from pipeline.pipeline import Pipeline
from pipeline.libs.motion_detector import MotionDetector


class DetectMotion(Pipeline):
    """Pipeline task to detect motion in video."""
    def __init__(self, p_min_motion_area):
        self.detector = MotionDetector(p_min_motion_area)
        super(DetectMotion, self).__init__()

    def generator(self):
        """Yields the image enriched with motion metadata"""
        stop = False
        while self.has_next() and not stop:
            try:
                # Buffer the pipeline stream
                data = next(self.source)
            except StopIteration:
                stop = True

            fg_img, bboxes = self.detector.detect(data["image"])
            data["motion_bboxes"] = bboxes

            if self.filter(data):
                yield self.map(data)

            cv2.imshow('frame', fg_img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break





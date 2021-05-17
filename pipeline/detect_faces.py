import numpy as np

from pipeline.pipeline import Pipeline
from pipeline.libs.face_detector import FaceDetector


class DetectFaces(Pipeline):
    """Pipeline task to detect faces from the image"""

    def __init__(self, prototxt, model, batch_size=1, confidence=0.5):
        self.detector = FaceDetector(prototxt, model, confidence=confidence)
        self.batch_size = batch_size

        super(DetectFaces, self).__init__()

    def generator(self):
        """Yields the image enriched with detected faces metadata"""

        stop = False
        while self.has_next() and not stop:
            try:
                # Buffer the pipeline stream
                data = next(self.source)
            except StopIteration:
                stop = True

            if data["motion_bboxes"]:
                faces = self.detector.detect(data["image"])
                data["faces"] = faces[0]
            else:
                data["faces"] = []

            # Yield all the data from buffer
            if self.filter(data):
                yield self.map(data)


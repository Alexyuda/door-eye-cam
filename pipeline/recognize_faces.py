import numpy as np
import os

from pipeline.pipeline import Pipeline
from pipeline.libs.face_embedder import FaceEmbedder


class RecognizeFaces(Pipeline):
    """Pipeline task to recognize faces patches"""

    def __init__(self, model, confidence=0.5):
        self.detector = FaceEmbedder(model=model)
        super(RecognizeFaces, self).__init__()

    def generator(self):
        """Yields the image enriched with detected faces label metadata"""

        stop = False
        while self.has_next() and not stop:
            try:
                # Buffer the pipeline stream
                data = next(self.source)
            except StopIteration:
                stop = True


            # Yield all the data from buffer
            if self.filter(data):
                yield self.map(data)


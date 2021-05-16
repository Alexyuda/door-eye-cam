import cv2
import datetime

from pipeline.pipeline import Pipeline
from pipeline.libs.colors import colors
from pipeline.libs.text import put_text


class AnnotateImage(Pipeline):
    """Pipeline task for image annotation."""

    def __init__(self, dst):
        self.dst = dst
        super(AnnotateImage, self).__init__()

    def map(self, data):
        data = self.annotate_faces(data)

        return data

    def annotate_faces(self, data):
        """Add annotations to image."""

        if "faces" not in data:  # in the case we switch off the face detector
            return data

        annotated_image = data["image"].copy()
        faces = data["faces"]
        motion_bboxes = data["motion_bboxes"]

        # add date time
        dt = str(datetime.datetime.now())
        put_text(annotated_image, dt, (0, 0),
                 color=colors.get("yellow1").to_bgr(),
                 bg_color=colors.get("green").to_bgr()
                 )

        # Loop over the faces and draw a rectangle around each
        for i, face in enumerate(faces):
            box, confidence = face
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), colors.get("green").to_bgr(), 2)
            put_text(annotated_image, f"{confidence:.2f}", (x1 - 1, y1),
                     color=colors.get("white").to_bgr(),
                     bg_color=colors.get("green").to_bgr(),
                     org_pos="bl")

        for i, box in enumerate(motion_bboxes):
            (x1, y1, x2, y2) = box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), colors.get("red").to_bgr(), 2)
            put_text(annotated_image, f"{i}", (x1 - 1, y1),
                     color=colors.get("white").to_bgr(),
                     bg_color=colors.get("green").to_bgr(),
                     org_pos="bl")

        data[self.dst] = annotated_image

        return data

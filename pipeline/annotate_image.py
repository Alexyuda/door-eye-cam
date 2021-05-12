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
        """Annotates faces on the image with bounding box and confidence info."""

        if "faces" not in data:  # in the case we switch off the face detector
            return data

        annotated_image = data["image"].copy()
        faces = data["faces"]

        # Loop over the faces and draw a rectangle around each
        for i, face in enumerate(faces):
            box, confidence = face
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), colors.get("green").to_bgr(), 2)
            put_text(annotated_image, f"{confidence:.2f}", (x1 - 1, y1),
                     color=colors.get("white").to_bgr(),
                     bg_color=colors.get("green").to_bgr(),
                     org_pos="bl")

            # add date time
            dt = str(datetime.datetime.now())
            put_text(annotated_image, dt, (0, 0),
                     color=colors.get("yellow1").to_bgr(),
                     bg_color=colors.get("green").to_bgr()
                     )

            # cv2.putText(annotated_image,
            #             text=dt,
            #             org=(10, 100),
            #             fontScale=1,
            #             color=colors.get("yellow").to_bgr(),
            #             thickness=1,
            #             lineType=cv2.LINE_8)

        data[self.dst] = annotated_image

        return data

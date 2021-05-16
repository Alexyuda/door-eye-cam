import cv2
import imutils


class MotionDetector:

    def __init__(self, p_min_area=0.05):
        self.fg_model = cv2.createBackgroundSubtractorMOG2(history=1500, detectShadows=False)
        self.p_min_area = p_min_area

    def detect(self, image):
        image = self.preprocess(image)
        fg_img = self.fg_model.apply(image)
        bboxes = self.extract_motion_bboxes(fg_img)
        return fg_img, bboxes

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        return gray

    def extract_motion_bboxes(self, image):
        image = cv2.dilate(image, None, iterations=2)
        height, width, = image.shape
        min_area = height * width * self.p_min_area
        cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        bboxes = []
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            bboxes.append((x, y, x + w, y + h))
        return bboxes

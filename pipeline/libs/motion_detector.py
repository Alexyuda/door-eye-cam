import cv2
import imutils
import numpy as np


class MotionDetector:

    kernel_open = np.ones((5, 5), np.uint8)
    kernel_dilate = np.ones((21, 21), np.uint8)

    def __init__(self, p_min_area=0.01):
        # self.fg_model = cv2.createBackgroundSubtractorMOG2(history=1500, varThreshold=16, detectShadows=False)
        self.fg_model = cv2.createBackgroundSubtractorKNN(history=1500, detectShadows=True)
        self.p_min_area = p_min_area

    def detect(self, image):
        image = self.preprocess(image)
        fg_img = self.fg_model.apply(image)
        fg_img = self.postproc(fg_img)
        bboxes = self.extract_motion_bboxes(fg_img)
        return fg_img, bboxes

    def preprocess(self, image):
        image = cv2.GaussianBlur(image, (21, 21), 0)
        return image

    def postproc(self, image):
        image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, self.kernel_open)
        image = cv2.dilate(image, self.kernel_dilate)
        return image

    def extract_motion_bboxes(self, image):
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

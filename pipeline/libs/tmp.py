from deepface import DeepFace
import time
from imutils import face_utils
import dlib
import cv2

shape_predictor = dlib.shape_predictor(r"C:\Repositories\door-eye-cam\models\FacialLandmarks\shape_predictor_5_face_landmarks.dat")
fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))

raw_img = cv2.imread(r"C:\Users\alexy\Desktop\Capture.JPG")
# convert to gray scale image

for _ in range(100):
    start = time.time()

    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    # align and resize
    aligned_face = fa.align(raw_img, gray, dlib.rectangle(left=1, top=1, right=gray.shape[1] - 1, bottom=gray.shape[0] - 1))
    aligned_face = cv2.resize(aligned_face, (112, 112))

    end = time.time()
    print(end - start)

cv2.imshow("", aligned_face)
cv2.waitKey(0)
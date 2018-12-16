import os,sys
import cv2
import dlib
from imutils import face_utils

cap = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
face_parts_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while True:
    tick = cv2.getTickCount()

    ret, rgb = cap.read()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_detector(gray)

    if len(faces) == 1:
        face = faces[0]
        cv2.rectangle(rgb, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

        face_parts = face_parts_detector(gray, face)
        face_parts = face_utils.shape_to_np(face_parts)

        for i, ((x, y)) in enumerate(face_parts[:]):
            cv2.circle(rgb, (x, y), 1, (0, 255, 0), -1)
            cv2.putText(rgb, str(i), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick)
    cv2.putText(rgb, "FPS:{} ".format(int(fps)), 
        (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', rgb)

    if cv2.waitKey(1) == 27:
        break  # esc to quit

cap.release()
cv2.destroyAllWindows()

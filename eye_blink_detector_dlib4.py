import os,sys
import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
face_parts_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def calc_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_ear = (A + B) / (2.0 * C)
    return round(eye_ear, 3)

def eye_marker(face_mat, position):
    for i, ((x, y)) in enumerate(position):
        cv2.circle(face_mat, (x, y), 1, (255, 255, 255), -1)
        cv2.putText(face_mat, str(i), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

while True:
    tick = cv2.getTickCount()

    ret, rgb = cap.read()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))

    if len(faces) == 1:
        x, y, w, h = faces[0, :]
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        face_gray = gray[y :(y + h), x :(x + w)]
        scale = 480 / h
        face_gray_resized = cv2.resize(face_gray, dsize=None, fx=scale, fy=scale)

        face = dlib.rectangle(0, 0, face_gray_resized.shape[1], face_gray_resized.shape[0])
        face_parts = face_parts_detector(face_gray_resized, face)
        face_parts = face_utils.shape_to_np(face_parts)

        left_eye = face_parts[42:48]
        eye_marker(face_gray_resized, left_eye)

        left_eye_ear = calc_ear(left_eye)
        cv2.putText(rgb, "LEFT eye EAR:{} ".format(left_eye_ear), 
            (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

        right_eye = face_parts[36:42]
        eye_marker(face_gray_resized, right_eye)

        right_eye_ear = calc_ear(right_eye)
        cv2.putText(rgb, "RIGHT eye EAR:{} ".format(round(right_eye_ear, 3)), 
            (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

        if (left_eye_ear + right_eye_ear) < 0.55:
            cv2.putText(rgb,"Sleepy eyes. Wake up!",
                (10,180), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3, 1)

        cv2.imshow('frame_resize', face_gray_resized)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick)
    cv2.putText(rgb, "FPS:{} ".format(int(fps)), 
        (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', rgb)
    if cv2.waitKey(1) == 27:
        break  # esc to quit

cap.release()
cv2.destroyAllWindows()

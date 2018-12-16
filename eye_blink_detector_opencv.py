import os,sys
import cv2

cap = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

while True:
    ret, rgb = cap.read()

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))

    if len(faces) == 1:
        x, y, w, h = faces[0, :]
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 処理高速化のために顔の上半分を検出対象範囲とする
        eyes_gray = gray[y : y + int(h/2), x : x + w]
        eyes = eye_cascade.detectMultiScale(
            eyes_gray, scaleFactor=1.11, minNeighbors=3, minSize=(8, 8))

        for ex, ey, ew, eh in eyes:
            cv2.rectangle(rgb, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 255, 0), 1)

        if len(eyes) == 0:
            cv2.putText(rgb,"Sleepy eyes. Wake up!",
                (10,100), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 2, 1)

    cv2.imshow('frame', rgb)
    if cv2.waitKey(1) == 27:
        break  # esc to quit

cap.release()
cv2.destroyAllWindows()

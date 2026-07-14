import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_eye.xml"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:

            eye_region = roi_color[ey:ey+eh, ex:ex+ew]

            b, g, r = cv2.split(eye_region)

            red_score = np.mean(r)

            color = (255, 0, 0)
            text = "Normal Eye"

            if red_score > 80:
                color = (0, 0, 255)
                text = "Possible Red Eye"

            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex+ew, ey+eh),
                color,
                2
            )

            cv2.putText(
                roi_color,
                text,
                (ex, ey-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

    cv2.imshow("Red Eye Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
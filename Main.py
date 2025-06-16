import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time


model = load_model("mask_model.h5")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("⚠️ Could not load Haar cascade. Make sure 'haarcascade_frontalface_default.xml' is in the same directory.")
    exit()


def preprocess_image(image, target_size=(100, 100)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, target_size)
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)


cap = cv2.VideoCapture(0)
last_alert_time = 0

print("(Press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        preprocessed = preprocess_image(face_img)

        prediction = model.predict(preprocessed)[0][0]
        label = "Mask" if prediction < 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        if label == "No Mask" and time.time() - last_alert_time > 5:
            print("🚨 Alert: No Mask Detected!")
            last_alert_time = time.time()

    cv2.imshow("Live Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

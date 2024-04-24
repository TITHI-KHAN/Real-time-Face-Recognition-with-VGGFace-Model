import cv2
import os
import numpy as np
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.preprocessing import image


model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features

known_faces = {}
for dir_name in os.listdir('training_data'):
    subject_path = os.path.join('training_data', dir_name)
    if not os.path.isdir(subject_path):
        continue

    face_images = os.listdir(subject_path)
    if face_images:
        features = extract_features(os.path.join(subject_path, face_images[0])) # Assuming one image per person for simplicity
        known_faces[dir_name] = features


video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_rect = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces_rect:
        face = cv2.resize(frame[y:y+h, x:x+w], (224, 224))
        face_array = np.asarray(face, dtype='float32')
        face_array = np.expand_dims(face_array, axis=0)
        face_array = preprocess_input(face_array)
        captured_features = model.predict(face_array)


        min_dist = float('inf')
        identity = None


        for name, features in known_faces.items():
            dist = cosine(features, captured_features)
            if dist < min_dist:
                min_dist = dist
                identity = name

        label_text = f"{identity if identity else 'Unknown'}, {min_dist:.2f}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    cv2.imshow("Real-time Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

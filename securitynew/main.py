import cv2
import numpy as np
import os
from playsound import playsound
from mtcnn import MTCNN

# Define your knn function for face classification
def knn(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])
    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    distances = np.array(dk)[:, 0]
    return labels, distances

def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with video path

# Initialize MTCNN for face detection
mtcnn_detector = MTCNN()

# Path to store face dataset
dataset_path = "./finalface/"

# Initialize lists for face data and labels
face_data = []
labels = []
class_id = 0
names = {}

# Load face data from dataset
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(os.path.join(dataset_path, fx))

        # Flatten data if it has more than 2 dimensions
        if data_item.ndim > 2:
            data_item = data_item.reshape(data_item.shape[0], -1)

        face_data.append(data_item)
        target = class_id * np.ones((data_item.shape[0],))
        labels.append(target)
        class_id += 1

# Prepare training data
face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

# Normalize face data
face_dataset = face_dataset / 255.0  # Normalize pixel values

trainset = np.concatenate((face_dataset, face_labels), axis=1)

# Font for displaying names on the video frame
font = cv2.FONT_HERSHEY_SIMPLEX

# Main loop for face detection and recognition
while True:
    # Start time for frame processing time measurement
    start_time = cv2.getTickCount()

    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces using MTCNN
    faces = mtcnn_detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']

        # Crop and resize face region
        face_section = frame[y:y+h, x:x+w]
        face_section = cv2.resize(face_section, (100, 100))
        face_section_flattened = face_section.flatten() / 255.0

        # Perform KNN classification
        pred_labels, distances = knn(trainset, face_section_flattened, k=7)

        # Get the predicted class and its distance
        pred_label = int(pred_labels[0])
        confidence = distances[0]

        # Display the name and bounding box on the video frame
        if pred_label in names and confidence < 60:
            cv2.putText(frame, names[pred_label], (x, y - 10), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            # Use absolute path for playsound
            playsound(r'C:\Users\Hasu\PycharmProjects\securitynew\beep-02.wav', block=False)
        else:
            cv2.putText(frame, "Unknown", (x, y - 10), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Calculate processing time and delay to match video frame rate
    end_time = cv2.getTickCount()
    processing_time = (end_time - start_time) / cv2.getTickFrequency()
    delay = max(1, int((1 / cap.get(cv2.CAP_PROP_FPS) - processing_time) * 1000))

    cv2.imshow("Video Frame", frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

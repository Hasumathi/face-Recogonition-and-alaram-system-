import cv2
import numpy as np
import os
from mtcnn import MTCNN

cap = cv2.VideoCapture(1)

# Initialize MTCNN for face detection
mtcnn_detector = MTCNN()

skip = 0
face_data = []
dataset_path = "./finalface/"

# Ensure the dataset path exists
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

file_name = input("Enter the name of person: ")

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    # Detect faces using MTCNN
    faces = mtcnn_detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']

        offset = 10
        face_offset = frame[max(y - offset, 0):y + h + offset, max(x - offset, 0):x + w + offset]

        # Ensure the face offset is within bounds
        if y - offset < 0 or y + h + offset > frame.shape[0] or x - offset < 0 or x + w + offset > frame.shape[1]:
            continue

        face_selection = cv2.resize(face_offset, (100, 100))

        if skip % 10 == 0:
            face_data.append(face_selection)
            print(f"Collected face data count: {len(face_data)}")

        cv2.imshow("Face Selection", face_selection)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Faces", frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# Convert face data to numpy array
face_data = np.array(face_data)
print(f"Face data shape: {face_data.shape}")

# Construct the full file path
file_path = os.path.join(dataset_path, file_name + '.npy')
print(f"Saving dataset to: {file_path}")

# Save the numpy array to the specified file path
np.save(file_path, face_data)
print(f"Dataset saved at: {file_path}")

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import os
import winsound
from mtcnn import MTCNN

def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())

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

# Initialize video capture and MTCNN face detector
video_paths = [0, 1]  # Indices for inbuilt and new webcam
caps = [cv2.VideoCapture(video_path) for video_path in video_paths]
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
        face_data.append(data_item)
        target = class_id * np.ones((data_item.shape[0],))
        labels.append(target)
        class_id += 1

# Prepare training data
face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

# Normalize face data
face_dataset_flattened = face_dataset.reshape(face_dataset.shape[0], -1)
face_dataset_flattened = face_dataset_flattened / 255.0  # Normalize pixel values

trainset = np.concatenate((face_dataset_flattened, face_labels), axis=1)

# Font for displaying names on the video frame
font = cv2.FONT_HERSHEY_SIMPLEX

# Main loop for face detection and recognition
while True:
    frames = []
    ret_list = []

    for cap in caps:
        ret, frame = cap.read()
        ret_list.append(ret)
        frames.append(frame)

    if not all(ret_list):
        print("Failed to capture frame from all cameras.")
        break

    for i, frame in enumerate(frames):
        if frame is None:
            print(f"Camera {i + 1}: No frame captured.")
            continue

        # Detect faces using MTCNN
        faces = mtcnn_detector.detect_faces(frame)

        # Debug: print number of faces detected in each frame
        print(f"Camera {i + 1}: {len(faces)} faces detected")

        for face in faces:
            x, y, w, h = face['box']
            keypoints = face['keypoints']

            # Align face using keypoints (optional)
            # Add your alignment logic here if needed

            offset = 10  # Increase offset for better face section capture
            face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]

            # Debug: Check face section dimensions
            print(f"Camera {i + 1}: Face section size: {face_section.shape}")

            if face_section.size == 0:
                print(f"Camera {i + 1}: Empty face section, skipping")
                continue

            face_section = cv2.resize(face_section, (100, 100))
            face_section_flattened = face_section.flatten()
            face_section_flattened = face_section_flattened / 255.0  # Normalize pixel values

            # Perform KNN classification
            pred_labels, distances = knn(trainset, face_section_flattened, k=7)

            # Get the predicted class and its distance
            pred_label = int(pred_labels[0])
            confidence = distances[0]

            # Display the name and bounding box on the video frame
            if pred_label in names and confidence < 50:  # Adjusted confidence threshold
                cv2.putText(frame, names[pred_label], (x, y - 10), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

                # Display the image of the detected person
                detected_image = cv2.imread(
                    os.path.join(dataset_path, f"{names[pred_label]}.jpg"))  # Assuming .jpg format
                if detected_image is not None:
                    cv2.imshow(f"Detected: {names[pred_label]}", detected_image)
                    cv2.moveWindow(f"Detected: {names[pred_label]}", 20, 20)

                winsound.Beep(1500, 500)  # Beep sound frequency and duration
                print(f"Camera {i + 1}: {names[pred_label]} detected with confidence {confidence:.2f}")
            else:
                print(f"Camera {i + 1}: Unknown face detected with confidence {confidence:.2f}")

        cv2.imshow(f"Camera {i + 1}", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()

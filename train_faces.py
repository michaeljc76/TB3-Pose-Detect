import cv2
import numpy as np
import os

faces_path = 'faces/'
faces = []
labels = []
names = {}

# Load training images
for i, name in enumerate(os.listdir(faces_path)):
    names[i] = name
    for img_name in os.listdir(f"{faces_path}/{name}"):
        img = cv2.imread(f"{faces_path}/{name}/{img_name}", 0)  # Read as grayscale
        img = cv2.resize(img, (100, 100))
        faces.append(img)
        labels.append(i)

# Train recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.write('face_recognizer.yml')

print(f"Trained {len(names)} people: {names}")
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
import cv2

# Create dummy directories
os.makedirs('dataset/Dummy_01', exist_ok=True)
os.makedirs('attendance_records', exist_ok=True)

# Generate dummy image files so the View Reports and Registration tabs look populated
dummy_img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
for i in range(1, 101):
    cv2.imwrite(f'dataset/Dummy_01/{i}.jpg', dummy_img)

# Generate a dummy model.pkl expecting 50x50 flattened images (2500 pixels)
faces_data = np.random.randint(0, 255, size=(100, 2500))
labels = ['Dummy_01'] * 100

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(faces_data, labels)
joblib.dump(knn, 'model.pkl')

print("Successfully generated dummy dataset and model.pkl!")

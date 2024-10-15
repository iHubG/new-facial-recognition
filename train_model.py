
import os
import pickle
import cv2
import numpy as np
from keras_facenet import FaceNet
from sklearn import svm
from sklearn.preprocessing import LabelEncoder

# Initialize the FaceNet embedder (this uses pre-trained weights)
embedder = FaceNet()

# Initialize lists to hold embeddings and labels
X = []
y = []

# Walk through dataset directory and process images
for root, dirs, files in os.walk('datasets'):
    for file in files:
        if file.endswith(('.jpg', '.png')):
            image_path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(image_path))  # Use folder name as label
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Extract face embeddings
            faces = embedder.extract(image, threshold=0.95)

            if len(faces) > 0:
                face_embedding = faces[0]['embedding']  # Take the first face detected
                X.append(face_embedding)
                y.append(label)

# Convert to numpy arrays
X = np.array(X)

# Label encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train SVM classifier
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(X, y_encoded)

# Save the trained SVM model
with open('svm_classifier_facenet.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Save the label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model training completed using pre-trained FaceNet embeddings and saved as 'svm_classifier_facenet.pkl'.")


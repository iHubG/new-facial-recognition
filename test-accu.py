import os
import cv2
import pickle
import numpy as np
from keras_facenet import FaceNet
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Load the saved SVM classifier
with open('svm_classifier_facenet.pkl', 'rb') as f:
    clf = pickle.load(f)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize FaceNet to extract embeddings
embedder = FaceNet()

# Path to the test dataset folder
test_data_folder = 'datasets'

# Store true labels and predicted labels
true_labels = []
predicted_labels = []

# Iterate through test data to calculate accuracy
for root, dirs, files in os.walk(test_data_folder):
    for file in files:
        if file.endswith(('.jpg', '.png')):
            image_path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(image_path))  # Folder name as label
            true_labels.append(label)

            # Load and process the image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Extract face embeddings using FaceNet
            faces = embedder.extract(image_rgb, threshold=0.95)

            if len(faces) > 0:
                test_embedding = faces[0]['embedding']  # Use the first face

                # Predict using the SVM model
                predicted_label_encoded = clf.predict([test_embedding])[0]

                # Decode the predicted label
                predicted_label = label_encoder.inverse_transform([predicted_label_encoded])[0]
                predicted_labels.append(predicted_label)
            else:
                predicted_labels.append("Unknown")

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print classification report (precision, recall, etc.)
print(classification_report(true_labels, predicted_labels))


'''
import os
import pickle
from deepface import DeepFace
from sklearn.metrics import accuracy_score, classification_report

# Load the saved SVM classifier
with open('svm_classifier_deepface.pkl', 'rb') as f:
    clf = pickle.load(f)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Path to the test dataset folder
test_data_folder = 'datasets'

# Store true labels and predicted labels
true_labels = []
predicted_labels = []

# Iterate through test data to calculate accuracy
for root, dirs, files in os.walk(test_data_folder):
    for file in files:
        if file.endswith(('.jpg', '.png')):
            image_path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(image_path))  # Folder name as label
            true_labels.append(label)

            # Extract embeddings using DeepFace
            try:
                embeddings = DeepFace.represent(image_path, model_name='VGG-Face', enforce_detection=False)
                if embeddings:  # Ensure embeddings are extracted
                    test_embedding = embeddings[0]['embedding']  # Use the first face
                    # Predict using the SVM model
                    predicted_label_encoded = clf.predict([test_embedding])[0]

                    # Decode the predicted label
                    predicted_label = label_encoder.inverse_transform([predicted_label_encoded])[0]
                    predicted_labels.append(predicted_label)
                else:
                    predicted_labels.append("Unknown")
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                predicted_labels.append("Unknown")

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Optionally, print classification report (precision, recall, etc.)
print(classification_report(true_labels, predicted_labels))
'''


'''
FaceNet
import os
import cv2
import pickle
import numpy as np
from keras_facenet import FaceNet
from sklearn.metrics import accuracy_score

# Load the saved SVM classifier
with open('svm_classifier_facenet.pkl', 'rb') as f:
    clf = pickle.load(f)

# Load the label encoder (optional if you used one)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize FaceNet to extract embeddings
embedder = FaceNet()

# Path to the test dataset folder
test_data_folder = 'datasets'

# Store true labels and predicted labels
true_labels = []
predicted_labels = []

# Iterate through test data to calculate accuracy
for root, dirs, files in os.walk(test_data_folder):
    for file in files:
        if file.endswith(('.jpg', '.png')):
            image_path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(image_path))  # Folder name as label
            true_labels.append(label)

            # Load and process the image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Extract face embeddings using FaceNet
            faces = embedder.extract(image_rgb, threshold=0.95)

            if len(faces) > 0:
                test_embedding = faces[0]['embedding']  # Use the first face

                # Predict using the SVM model
                predicted_label_encoded = clf.predict([test_embedding])[0]

                # Decode the predicted label
                predicted_label = label_encoder.inverse_transform([predicted_label_encoded])[0]
                predicted_labels.append(predicted_label)
            else:
                predicted_labels.append("Unknown")

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Optionally, print classification report (precision, recall, etc.)
from sklearn.metrics import classification_report
print(classification_report(true_labels, predicted_labels))
'''
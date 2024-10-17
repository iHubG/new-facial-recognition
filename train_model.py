import os
import pickle
import cv2
import numpy as np
from keras_facenet import FaceNet
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

# Initialize the FaceNet embedder
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

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train SVM classifier
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# Save the trained SVM model and label encoder
with open('svm_classifier_facenet.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model training completed using FaceNet embeddings and saved as 'svm_classifier_facenet.pkl'.")

# Evaluate the model on the validation set
y_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Calculate probabilities for threshold selection
y_proba = clf.predict_proba(X_val)

# Define thresholds to test
thresholds = np.arange(0.0, 1.0, 0.05)
best_threshold = 0.0
best_f1 = 0.0

# Iterate through thresholds to find the best one
for threshold in thresholds:
    y_pred_thresholded = np.argmax(y_proba, axis=1)  # Get the index of the max probability
    y_pred_confident = np.where(np.max(y_proba, axis=1) >= threshold, y_pred_thresholded, -1)  # -1 for unknown
    
    # Calculate precision, recall, f1-score
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred_confident, average='macro', zero_division=0)

    # Check if this is the best F1 score
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best Threshold: {best_threshold:.2f} with F1 Score: {best_f1:.2f}")

# Print the classification report with the best threshold applied
y_pred_best = np.where(np.max(y_proba, axis=1) >= best_threshold, np.argmax(y_proba, axis=1), -1)
print("\nClassification Report with Best Threshold:")
print(classification_report(y_val, y_pred_best, target_names=label_encoder.inverse_transform(np.unique(y_encoded))))

# Check the number of 'Unknown' faces detected using the best threshold
y_unknown = (y_proba.max(axis=1) < best_threshold)  # Identify low-confidence faces
print(f"Number of 'Unknown' faces detected in validation set: {y_unknown.sum()}")





'''
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
'''

from sklearn.datasets import fetch_lfw_people

# Download the LFW dataset (if not already cached) and load it
lfw_data = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Flatten the images for SVM input
X = lfw_data.data  # Flattened image data
y = lfw_data.target  # Target labels

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM model
svm_clf = SVC(kernel='linear', probability=True)
svm_clf.fit(X_train, y_train)

# Make predictions
y_pred = svm_clf.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=lfw_data.target_names))

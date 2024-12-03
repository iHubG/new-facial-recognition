import pickle

# Load the label_encoder.pkl file
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Inspect the type of the object for label encoder
print("Label Encoder Object:")
print(f"Type of loaded object: {type(label_encoder)}")
print(label_encoder)

# Check if it's a LabelEncoder
if hasattr(label_encoder, 'classes_'):
    print(f"Classes in the LabelEncoder: {label_encoder.classes_}")
else:
    print("The object is not a LabelEncoder.")

# Load the svm_classifier_facenet.pkl file
with open('svm_classifier_facenet.pkl', 'rb') as f:
    svm_classifier = pickle.load(f)

# Inspect the type of the object for the SVM classifier
print("\nSVM Classifier Object:")
print(f"Type of loaded object: {type(svm_classifier)}")

# Check if it has SVM attributes
if hasattr(svm_classifier, 'support_'):
    print(f"Support vectors: {svm_classifier.support_}")
    print(f"Classes: {svm_classifier.classes_}")
else:
    print("The object is not an SVM classifier, it might be a different type.")

# Optionally, print the classifier details
print(svm_classifier)

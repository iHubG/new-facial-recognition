import pickle

# Load the .pkl file
with open('svm_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

# Inspect the type of the object
print(f"Type of loaded object: {type(clf)}")

# If it's an object like an SVM classifier, you can inspect its attributes
if hasattr(clf, 'support_'):
    print("Classifier attributes:")
    print(f"Support vectors: {clf.support_}")
    print(f"Classes: {clf.classes_}")
else:
    print("The object is not an SVM classifier, it's a different type.")

# You can also inspect a few other parts of the object (depends on the type)
print(clf)

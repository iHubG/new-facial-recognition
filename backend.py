# flask_backend.py

from flask import Flask, request, jsonify
import face_recognition
import pickle
import cv2
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the SVM classifier
with open('svm_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

# Define the route for face recognition
@app.route('/recognize', methods=['POST'])
def recognize():
    # Read the image from the request
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Convert the image to RGB
    rgb_image = image[:, :, ::-1]

    # Detect and encode the faces
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    # Initialize an empty result list
    results = []

    # Predict the identity of each face
    for face_encoding in face_encodings:
        name = clf.predict([face_encoding])[0]
        results.append(name)

    return jsonify({"recognized": results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

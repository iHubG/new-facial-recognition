import cv2
import face_recognition
import pickle

# Load the trained SVM classifier
with open('svm_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

# Try opening the default webcam (camera index 0)
video_capture = cv2.VideoCapture(0)

# Check if the webcam was successfully opened
if not video_capture.isOpened():
    print("Error: Unable to access the webcam.")
else:
    print("Webcam accessed successfully!")

while True:
    # Capture frame from webcam
    ret, frame = video_capture.read()

    # Check if frame was captured successfully
    if not ret:
        print("Failed to capture frame.")
        break

    # Resize the frame if necessary (to ensure it fits within screen size)
    frame_resized = cv2.resize(frame, (640, 480))

    # Convert BGR frame to RGB for face_recognition
    rgb_frame = frame_resized[:, :, ::-1]  # Convert BGR to RGB
    
    # Detect faces using face_recognition (try using "hog" model)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

    # Debugging: print the face locations to check if the detection is happening
    print(f"Detected faces at locations: {face_locations}")

    # If faces are detected, extract face encodings and make predictions
    if len(face_locations) > 0:
        try:
            # Correctly extract face encodings using the face locations
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            print(f"Extracted face encodings: {face_encodings}")  # Debugging line
            
            # Predict the names of detected faces
            for face_encoding in face_encodings:
                name = clf.predict([face_encoding])
                print(f"Detected: {name[0]}")
                
            # Draw rectangles around detected faces
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame_resized, (left, top), (right, bottom), (0, 255, 0), 2)

        except Exception as e:
            print(f"Error extracting face encodings: {e}")
    else:
        print("No faces detected")

    # Display the frame with bounding boxes around faces
    cv2.imshow('Camera Feed with Face Detection', frame_resized)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any open windows
video_capture.release()
cv2.destroyAllWindows()

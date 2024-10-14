from flask import Blueprint, render_template, Response, jsonify, request, redirect, url_for
import cv2
import sqlite3
from datetime import datetime, timedelta
import threading
import pickle
import numpy as np
from keras_facenet import FaceNet
from sklearn import svm

# Create a Blueprint for face recognition
faceRecognition_bp = Blueprint('faceRecognition', __name__)

# Initialize variables for last insertion time and interval
last_insert_time = datetime.min
insert_interval = timedelta(minutes=5)
detected_info = {"name": None, "datetime": None}  # Store both name and timestamp

# Load the trained SVM classifier and label encoder
with open('svm_classifier_facenet.pkl', 'rb') as f:
    clf = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize FaceNet
embedder = FaceNet()

def insert_data(name, entry_datetime, period):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (name, entry_datetime, period) VALUES (?, ?, ?)", (name, entry_datetime, period))
        conn.commit()
    except Exception as e:
        print(f"Error inserting data: {e}")
    finally:
        conn.close()

def get_user_data(name):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT name, entry_datetime FROM users WHERE name = ?", (name,))
        result = c.fetchone()
    except Exception as e:
        print(f"Error fetching user data: {e}")
        result = None
    finally:
        conn.close()
    return result if result else (None, None)

class VideoCamera:
    def __init__(self):
        self.video = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.process_frame_interval = 5  # Process every nth frame
        self.frame_count = 0

    def start(self):
        if not self.running:
            self.video = cv2.VideoCapture(0)
            self.running = True
            self.frame_count = 0
            self.thread = threading.Thread(target=self.update_frame, daemon=True)
            self.thread.start()
            print("Camera started.")

    def stop(self):
        if self.running:
            self.running = False
            self.thread.join()  # Wait for the thread to finish
            self.video.release()
            print("Camera turned off.")

    def update_frame(self):
        while self.running:
            success, frame = self.video.read()
            if success:
                with self.lock:
                    self.frame = frame
                    self.frame_count += 1
            else:
                print("Failed to capture frame")

    def get_frame(self):
        with self.lock:
            return self.frame

    @property
    def is_running(self):
        return self.running

# Camera instance
camera = VideoCamera()

# Initialize the camera state
camera_running = False

@faceRecognition_bp.route('/faceRecognition', methods=['GET', 'POST'])
def dashboard():
    global camera_running

    if request.method == 'POST':
        if not camera_running:
            camera.start()
            camera_running = True

    return render_template('views/faceRecognition.html', camera_running=camera_running)

@faceRecognition_bp.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_running

    if camera_running:
        camera.stop()
    else:
        camera.start()
    
    camera_running = not camera_running
    return redirect(url_for('faceRecognition.dashboard'))

def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

@faceRecognition_bp.route('/get_all_users', methods=['GET'])
def get_all_users():
    conn = get_db_connection()
    users = conn.execute('SELECT * FROM users').fetchall()
    conn.close()
    
    users_list = [dict(user) for user in users]
    return jsonify(users_list)

def generate_frames():
    global last_insert_time, detected_info
    while camera.running:
        frame = camera.get_frame()
        if frame is None:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract face embeddings
        faces = embedder.extract(rgb_frame, threshold=0.95)

        detected_info["name"] = "Unknown"  # Default to Unknown
        for res in faces:
            face_embedding = res['embedding']
            face_embedding = np.array(face_embedding).reshape(1, -1)

            # Predict using SVM
            prediction = clf.predict(face_embedding)
            predicted_class_index = prediction[0]

            # Get the confidence score (distance from the decision boundary)
            confidence_scores = clf.decision_function(face_embedding)
            confidence = confidence_scores[0][predicted_class_index] if len(confidence_scores.shape) > 1 else confidence_scores[0]

            # Set confidence threshold (tweak as necessary)
            confidence_threshold = 0.5  # Adjust based on your needs
            distance_threshold = 1.0  # Distance threshold for unknown

            if confidence < confidence_threshold:
                name = "Unknown"
            else:
                # Get the actual name
                name = label_encoder.inverse_transform([predicted_class_index])[0]

            # Optionally compute a distance from the predicted embedding to the known ones
            # Here, you would need to implement logic to calculate this distance

            # Draw bounding box and name
            x, y, w, h = res['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

            # Update detected info
            detected_info["name"] = name if name != "Unknown" else detected_info["name"]

        # Encode and yield the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@faceRecognition_bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@faceRecognition_bp.route('/get_detected_info')
def get_detected_info():
    return jsonify(detected_info)

'''
from flask import Blueprint, render_template, Response, jsonify, request, redirect, url_for
import cv2
import sqlite3
from datetime import datetime, timedelta
import threading
import pickle
import numpy as np
from keras_facenet import FaceNet
from sklearn import svm

# Create a Blueprint for face recognition
faceRecognition_bp = Blueprint('faceRecognition', __name__)

# Initialize variables for last insertion time and interval
last_insert_time = datetime.min
insert_interval = timedelta(minutes=5)
detected_info = {"name": None, "datetime": None}  # Store both name and timestamp

# Load the trained SVM classifier and label encoder
with open('svm_classifier_facenet.pkl', 'rb') as f:
    clf = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize FaceNet
embedder = FaceNet()

def insert_data(name, entry_datetime, period):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO users (name, entry_datetime, period) VALUES (?, ?, ?)", (name, entry_datetime, period))
    conn.commit()
    conn.close()

def get_user_data(name):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT name, entry_datetime FROM users WHERE name = ?", (name,))
    result = c.fetchone()
    conn.close()
    return result if result else (None, None)  # Return both name and datetime

class VideoCamera:
    def __init__(self):
        self.video = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.process_frame_interval = 5  # Process every nth frame
        self.frame_count = 0

    def start(self):
        if not self.running:
            # Initialize video capture
            self.video = cv2.VideoCapture(0)
            self.running = True
            self.frame_count = 0
            self.thread = threading.Thread(target=self.update_frame, daemon=True)  # Set thread as daemon
            self.thread.start()
            print("Camera started.")

    def stop(self):
        if self.running:
            self.running = False
            self.thread.join()  # Wait for the thread to finish
            self.video.release()  # Release the camera
            print("Camera turned off.")

    def update_frame(self):
        while self.running:
            success, frame = self.video.read()
            if success:
                with self.lock:
                    self.frame = frame
                    self.frame_count += 1
            else:
                print("Failed to capture frame")

    def get_frame(self):
        with self.lock:
            return self.frame

    @property
    def is_running(self):
        return self.running  # Return the running state of the camera



# Camera instance
camera = VideoCamera()

# Initialize the camera state
camera_running = False  # Camera starts off

# Existing VideoCamera class and functions...

@faceRecognition_bp.route('/faceRecognition', methods=['GET', 'POST'])
def dashboard():
    global camera_running  # Declare camera_running as global

    if request.method == 'POST':
        if not camera_running:  # Only start the camera if it's not already running
            camera.start()  # Start the camera on form submit
            camera_running = True  # Update the camera state

    # Check if the camera is currently running
    return render_template('views/faceRecognition.html', camera_running=camera_running)

@faceRecognition_bp.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_running  # Declare camera_running as global

    if camera_running:
        camera.stop()  # Stop the camera
    else:
        camera.start()  # Start the camera
    
    camera_running = not camera_running  # Toggle the camera state
    return redirect(url_for('faceRecognition.dashboard'))  # Redirect back to the dashboard



def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

@faceRecognition_bp.route('/get_all_users', methods=['GET'])
def get_all_users():
    conn = get_db_connection()
    users = conn.execute('SELECT * FROM users').fetchall()
    conn.close()
    
    # Convert rows to a list of dictionaries
    users_list = [dict(user) for user in users]
    return jsonify(users_list)

def generate_frames():
    global last_insert_time, detected_info
    while camera.running:
        frame = camera.get_frame()
        if frame is None:
            continue

        camera.frame_count += 1
        if camera.frame_count % camera.process_frame_interval != 0:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract face embeddings
        faces = embedder.extract(rgb_frame, threshold=0.95)

        # Store names and detected info
        face_names = []
        for res in faces:
            face_embedding = res['embedding']
            # Predict the class label using the trained SVM classifier
            face_embedding = np.array(face_embedding).reshape(1, -1)  # Reshape for prediction
            prediction = clf.predict(face_embedding)
            name = label_encoder.inverse_transform(prediction)[0]

            face_names.append(name)

            # Retrieve user data from the database if the name is recognized
            retrieved_name, entry_datetime = get_user_data(name)
            if retrieved_name:
                detected_info["name"] = retrieved_name
                detected_info["datetime"] = entry_datetime
            else:
                detected_info["name"] = "Unknown"
                detected_info["datetime"] = None

            # Insert data into the database if it's time to do so
            now = datetime.now()
            entry_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
            period = now.strftime("%p")

            if (now - last_insert_time) > insert_interval:
                insert_data(name, entry_datetime, period)
                last_insert_time = now

            # Draw bounding box around the face
            x, y, w, h = res['box']  # Get the bounding box coordinates
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)  # Put name above the rectangle

        # Encode the frame to send to the browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@faceRecognition_bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@faceRecognition_bp.route('/get_detected_info')
def get_detected_info():
    return jsonify(detected_info)

# Start listening for commands in a separate thread (optional)
# command_thread = threading.Thread(target=listen_for_commands)
# command_thread.start()
'''
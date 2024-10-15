from flask import Blueprint, render_template, Response, jsonify, request, redirect, url_for
import cv2
import sqlite3
from datetime import datetime, timedelta
import threading
import pickle
import numpy as np
from keras_facenet import FaceNet
from sklearn import svm
import os

# Create a Blueprint for face recognition
faceRecognition_bp = Blueprint('faceRecognition', __name__)

# Initialize variables for last insertion time and interval
insert_interval = timedelta(minutes=5)
detected_info = {"name": None, "datetime": None, "grade_level": None, "section": None}  # Store additional info

# Load the trained SVM classifier and label encoder
with open('svm_classifier_facenet.pkl', 'rb') as f:
    clf = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize FaceNet
embedder = FaceNet()

def get_db_connection():
    conn = sqlite3.connect('face-recognition.db')
    conn.row_factory = sqlite3.Row
    return conn

def insert_data(name, entry_datetime, period, grade_level, section):
    try:
        print(f"Inserting data: {name}, {entry_datetime}, {period}, {grade_level}, {section}")
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("INSERT INTO users (name, entry_datetime, period, grade_level, section) VALUES (?, ?, ?, ?, ?)", 
                  (name, entry_datetime, period, grade_level, section))
        conn.commit()
        print(f"Successfully inserted data for {name}.")
    except Exception as e:
        print(f"Error inserting data: {e}")
    finally:
        conn.close()

def get_user_data(name):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT grade_level, section FROM users WHERE name = ?", (name,))
        result = c.fetchone()
        return {
            'grade_level': result[0],
            'section': result[1]
        } if result else None
    except Exception as e:
        print(f"Error fetching user data: {e}")
        return None
    finally:
        conn.close()

def get_latest_entry_datetime(name):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        # Retrieve the latest entry_datetime for the recognized name
        c.execute("SELECT entry_datetime FROM users WHERE name = ? ORDER BY entry_datetime DESC LIMIT 1", (name,))
        result = c.fetchone()
        return result[0] if result else None
    except Exception as e:
        print(f"Error fetching latest entry datetime: {e}")
        return None
    finally:
        conn.close()


class VideoCamera:
    def __init__(self):
        self.video = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        if not self.running:
            self.video = cv2.VideoCapture(0)
            self.running = True
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
            else:
                print("Failed to capture frame")

    def get_frame(self):
        with self.lock:
            return self.frame

# Camera instance
camera = VideoCamera()
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

@faceRecognition_bp.route('/get_all_users', methods=['GET'])
def get_all_users():
    conn = get_db_connection()
    users = conn.execute('SELECT * FROM users').fetchall()
    conn.close()
    
    users_list = [dict(user) for user in users]
    return jsonify(users_list)

# Initialize a dictionary to store the last insertion time for each user
last_insertion_times = {}

def generate_frames():
    global detected_info

    # Initialize a variable to hold the last valid detected information
    last_valid_detection = detected_info.copy()

    while camera.running:
        frame = camera.get_frame()
        if frame is None:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract face embeddings
        faces = embedder.extract(rgb_frame, threshold=0.95)

        # Default detected info
        current_detection = {
            "name": "Unknown",
            "datetime": None,
            "grade_level": None,
            "section": None
        }

        for res in faces:
            face_embedding = res['embedding']
            face_embedding = np.array(face_embedding).reshape(1, -1)

            # Predict using SVM
            prediction = clf.predict(face_embedding)
            predicted_class_index = prediction[0]

            # Get the confidence score
            confidence_scores = clf.decision_function(face_embedding)
            confidence = confidence_scores[0][predicted_class_index] if len(confidence_scores.shape) > 1 else confidence_scores[0]

            # Set confidence threshold
            confidence_threshold = 0.5

            if confidence < confidence_threshold:
                name = "Unknown"
            else:
                name = label_encoder.inverse_transform([predicted_class_index])[0]

            # Draw bounding box and name
            x, y, w, h = res['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

            # Current date and time
            now = datetime.now()
            entry_datetime = now.strftime("%m/%d/%Y %I:%M:%S")  # 12-hour format with AM/PM

            # Check if the user is recognized
            retrieved_name = name if name != "Unknown" else None

            if retrieved_name:
                current_detection["name"] = retrieved_name

                # Look for the user's corresponding folder
                grade_level, section = None, None

                for root, dirs, files in os.walk('datasets'):
                    for dir in dirs:
                        section_path = os.path.join(root, dir)
                        for student_file in os.listdir(section_path):
                            if retrieved_name in student_file:
                                grade_level = os.path.basename(os.path.dirname(section_path))
                                section = dir
                                break
                        if grade_level and section:
                            break

                current_detection["grade_level"] = grade_level if grade_level else "Unknown"
                current_detection["section"] = section if section else "Unknown"

                # Update datetime immediately after recognition
                current_detection["datetime"] = entry_datetime + " " + now.strftime("%p")  # Include AM/PM

                # Insert data into the database if the interval has passed
                if retrieved_name not in last_insertion_times:
                    last_insertion_times[retrieved_name] = now - insert_interval  # Set initial time

                if now - last_insertion_times[retrieved_name] >= insert_interval:
                    insert_data(retrieved_name, entry_datetime, now.strftime("%p"), current_detection["grade_level"], current_detection["section"])
                    last_insertion_times[retrieved_name] = now  # Update last insertion time

        # Update detected_info with current detection or maintain last valid detection
        if current_detection["name"] != "Unknown":
            detected_info.update(current_detection)
            last_valid_detection = current_detection  # Update last valid detection
        else:
            # Maintain previous valid detection information
            detected_info.update(last_valid_detection)

        # Print detected information for debugging
        print(f"Detected: {detected_info['name']}, Time: {detected_info['datetime']}, Grade: {detected_info['grade_level']}, Section: {detected_info['section']}")

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
insert_interval = timedelta(minutes=5)
detected_info = {"name": None, "datetime": None}  # Store both name and timestamp

# Load the trained SVM classifier and label encoder
with open('svm_classifier_facenet.pkl', 'rb') as f:
    clf = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize FaceNet
embedder = FaceNet()

def get_db_connection():
    conn = sqlite3.connect('face-recognition.db')
    conn.row_factory = sqlite3.Row
    return conn

def insert_data(name, entry_datetime, period):
    try:
        print(f"Inserting data: {name}, {entry_datetime}, {period}")
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("INSERT INTO users (name, entry_datetime, period) VALUES (?, ?, ?)", (name, entry_datetime, period))
        conn.commit()
        print(f"Successfully inserted data for {name}.")
    except Exception as e:
        print(f"Error inserting data: {e}")
    finally:
        conn.close()

def get_user_data(name):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT name FROM users WHERE name = ?", (name,))
        result = c.fetchone()
        return result[0] if result else None
    except Exception as e:
        print(f"Error fetching user data: {e}")
        return None
    finally:
        conn.close()

def get_latest_entry_datetime(name):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT entry_datetime FROM users WHERE name = ? ORDER BY entry_datetime DESC LIMIT 1", (name,))
        result = c.fetchone()
        return result[0] if result else None
    except Exception as e:
        print(f"Error fetching latest entry datetime: {e}")
        return None
    finally:
        conn.close()

class VideoCamera:
    def __init__(self):
        self.video = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        if not self.running:
            self.video = cv2.VideoCapture(0)
            self.running = True
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
            else:
                print("Failed to capture frame")

    def get_frame(self):
        with self.lock:
            return self.frame

# Camera instance
camera = VideoCamera()
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

@faceRecognition_bp.route('/get_all_users', methods=['GET'])
def get_all_users():
    conn = get_db_connection()
    users = conn.execute('SELECT * FROM users').fetchall()
    conn.close()
    
    users_list = [dict(user) for user in users]
    return jsonify(users_list)

# Initialize a dictionary to store the last insertion time for each user
last_insertion_times = {}

def generate_frames():
    global detected_info

    while camera.running:
        frame = camera.get_frame()
        if frame is None:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract face embeddings
        faces = embedder.extract(rgb_frame, threshold=0.95)

        # Default detected info
        detected_info["name"] = "Unknown"
        detected_info["datetime"] = None

        for res in faces:
            face_embedding = res['embedding']
            face_embedding = np.array(face_embedding).reshape(1, -1)

            # Predict using SVM
            prediction = clf.predict(face_embedding)
            predicted_class_index = prediction[0]

            # Get the confidence score
            confidence_scores = clf.decision_function(face_embedding)
            confidence = confidence_scores[0][predicted_class_index] if len(confidence_scores.shape) > 1 else confidence_scores[0]

            # Set confidence threshold
            confidence_threshold = 0.5

            if confidence < confidence_threshold:
                name = "Unknown"
            else:
                name = label_encoder.inverse_transform([predicted_class_index])[0]

            # Draw bounding box and name
            x, y, w, h = res['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

            # Current date and time
            now = datetime.now()
            entry_datetime = now.strftime("%Y/%m/%d %I:%M:%S")  # 12-hour format with AM/PM

            # Check if the user is recognized
            retrieved_name = name if name != "Unknown" else None

            if retrieved_name:
                detected_info["name"] = retrieved_name  # Store the recognized name

                # Check the last insertion time for the recognized user
                if retrieved_name not in last_insertion_times:
                    last_insertion_times[retrieved_name] = now - insert_interval  # Set initial time
                
                # Insert data into the database if the interval has passed
                if now - last_insertion_times[retrieved_name] >= insert_interval:
                    period = now.strftime("%p")  # e.g., AM/PM
                    insert_data(retrieved_name, entry_datetime, period)
                    last_insertion_times[retrieved_name] = now  # Update last insertion time

                # Retrieve the latest entry datetime and update detected_info
                latest_entry_datetime = get_latest_entry_datetime(retrieved_name)
                detected_info["datetime"] = latest_entry_datetime  # Set the latest time
            else:
                detected_info["name"] = "Unknown"
                detected_info["datetime"] = None

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

from flask import Blueprint, render_template, Response, jsonify, request, redirect, url_for, send_from_directory
import cv2
import sqlite3
from datetime import datetime, timedelta
import threading
import pickle
import numpy as np
from keras_facenet import FaceNet
from sklearn import svm
import os
import dlib
import time
import logging
from collections import deque
import requests

# Initialize logging
logging.basicConfig(filename='face_recognition.log', level=logging.INFO)

# Example of logging predictions
def log_prediction(image_path, expected, predicted):
    logging.info(f"Image: {image_path}, Expected: {expected}, Predicted: {predicted}")

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

        # Check if the entry already exists for the same user on the same day
        c.execute("""
            SELECT COUNT(*) 
            FROM users 
            WHERE name = ? AND entry_datetime LIKE ?
        """, (name, entry_datetime.split(" ")[0] + '%'))  # Check only the date part

        exists = c.fetchone()[0]

        print(f"Exists check: {exists} for {name} on {entry_datetime}")

        if exists == 0:
            # Insert the new entry
            c.execute("""
                INSERT INTO users (name, entry_datetime, period, grade_level, section) 
                VALUES (?, ?, ?, ?, ?)
            """, (name, entry_datetime, period, grade_level, section))
            conn.commit()
            print(f"Successfully inserted data for {name}.")
        else:
            print(f"Data for {name} already exists for this day. No insertion made.")
    
    except Exception as e:
        print(f"Error inserting data: {e}")
    finally:
        conn.close()
        
def async_insert_data(name, entry_datetime, period, grade_level, section):
    threading.Thread(target=insert_data, args=(name, entry_datetime, period, grade_level, section)).start()
    
from datetime import datetime, timedelta

def upsert_attendance(name, grade_level, section):
    try:
        conn = get_db_connection()
        c = conn.cursor()

        # Count total attendance from the users table (all entries)
        c.execute("""
            SELECT COUNT(*) AS total_attendance 
            FROM users 
            WHERE name = ?
        """, (name,))
        total_attendance = c.fetchone()[0] or 0  # Default to 0 if None

        # Count weekly attendance from the users table (all entries in the current week)
        start_of_week = datetime.now() - timedelta(days=datetime.now().weekday())
        c.execute("""
            SELECT COUNT(*) AS weekly_attendance 
            FROM users 
            WHERE name = ? AND entry_datetime >= ?
        """, (name, start_of_week.strftime('%m/%d/%Y 00:00:00')))
        weekly_attendance = c.fetchone()[0] or 0  # Default to 0 if None

        # Insert or update the attendance record based on the counts
        c.execute("""
            INSERT INTO attendance (name, grade_level, section, total_attendance, weekly_attendance, week)
            VALUES (?, ?, ?, ?, ?, strftime('%W', 'now'))
            ON CONFLICT(name) DO UPDATE SET
                total_attendance = ?,
                weekly_attendance = ?,
                week = strftime('%W', 'now')
        """, (name, grade_level, section, total_attendance, weekly_attendance, total_attendance, weekly_attendance))

        conn.commit()
        print(f"Successfully updated attendance for {name}.")
    except Exception as e:
        print(f"Error updating attendance data: {e}")
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
            self.video = cv2.VideoCapture(1)
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

@faceRecognition_bp.route('/get_all_users_data/<string:name>', methods=['GET'])
def get_user_data(name):
    conn = get_db_connection()
    query = '''
        SELECT name, entry_datetime, period, section, grade_level 
        FROM users 
        WHERE name = ? 
        ORDER BY id DESC  
        LIMIT 5
    '''
    users = conn.execute(query, (name,)).fetchall()
    conn.close()
    
    return jsonify([dict(user) for user in users]) # Return user data as a list of dictionaries

def async_fetch_user_data(name):
    try:
        response = requests.get(f'http://127.0.0.1:5000/user/get_all_users_data/{name}')
        if response.ok:
            user_data = response.json()
            # Process user_data here if needed
        else:
            print("Failed to fetch user data")
    except Exception as e:
        print(f"Error fetching user data: {e}")

@faceRecognition_bp.route('/datasets/<path:filepath>')
def serve_image(filepath):
    datasets_dir = 'datasets'
    return send_from_directory(datasets_dir, filepath, as_attachment=False)

@faceRecognition_bp.route('/attendance/<string:name>', methods=['GET'])
def get_attendance(name):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get attendance data from the attendance table
    cursor.execute("SELECT total_attendance, weekly_attendance FROM attendance WHERE name = ?", (name,))
    attendance_data = cursor.fetchone()

    # Set defaults if no data found
    if attendance_data:
        total_attendance = attendance_data['total_attendance']
        weekly_attendance = attendance_data['weekly_attendance']
    else:
        total_attendance = 0
        weekly_attendance = 0

    conn.close()

    return jsonify({
        'total_attendance': total_attendance,
        'weekly_attendance': weekly_attendance
    })
    
def get_user_picture(name, grade_level, section):
    """
    Fetch the first picture of the recognized user.
    """
    user_folder = os.path.join('datasets', grade_level, section, name)
    if os.path.isdir(user_folder):
        images = [f for f in os.listdir(user_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            # Return the relative path for the image
            return f'{grade_level}/{section}/{name}/{images[0]}'
    return None  # Return None if no images found or user folder does not exist

# Initialize a dictionary to store the last insertion time for each user
last_insertion_times = {}

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmark points for detecting blinks
LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))

# Helper function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def is_blinking(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return False  # No face detected

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in LEFT_EYE_POINTS])
        right_eye = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in RIGHT_EYE_POINTS])

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Set threshold for blinking based on EAR
        blink_threshold = 0.20

        if left_ear < blink_threshold and right_ear < blink_threshold:
            return True  # Blink detected

    return False  # No blink detected

# Initialize parameters for smoothing the bounding box
bounding_box_history = deque(maxlen=5)  # Store the last 5 bounding box positions

def smooth_bounding_box(current_box):
    """Smooth the bounding box position."""
    bounding_box_history.append(current_box)  # Add current box to history
    if len(bounding_box_history) > 0:
        # Calculate the average box coordinates
        avg_x = int(np.mean([box[0] for box in bounding_box_history]))
        avg_y = int(np.mean([box[1] for box in bounding_box_history]))
        avg_w = int(np.mean([box[2] for box in bounding_box_history]))
        avg_h = int(np.mean([box[3] for box in bounding_box_history]))
        return (avg_x, avg_y, avg_w, avg_h)
    return current_box

# Dictionary to store recognized users for the session
recognized_users = {}

current_user_name = "Unknown"

def generate_frames():
    global detected_info, current_user_name
    last_valid_detection = detected_info.copy()
    last_recognition_time = time.time()  # Initialize the last recognition time
    recognition_cooldown = 2  # Set cooldown period in seconds

    while camera.running:
        frame = camera.get_frame()
        if frame is None:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Check if a blink is detected to validate live face
        if is_blinking(frame):
            faces = embedder.extract(rgb_frame, threshold=0.95)

            current_detection = {
                "name": "Unknown",
                "datetime": None,
                "grade_level": None,
                "section": None,
                "confidence": 0.0  # To store the confidence score
            }

            if len(faces) > 0:
                current_time = time.time()  # Get the current time

                # Process the first detected face
                res = faces[0]
                face_embedding = np.array(res['embedding']).reshape(1, -1)

                # Predict using SVM and get the probability predictions
                probabilities = clf.predict_proba(face_embedding)
                max_prob_index = np.argmax(probabilities)
                max_prob = probabilities[0][max_prob_index]

                # Set confidence threshold for unknown faces (e.g., 0.5)
                confidence_threshold = 0.6

                name = label_encoder.inverse_transform([max_prob_index])[0] if max_prob >= confidence_threshold else "Unknown"

                # Prevent showing "Unknown" after a successful recognition
                if name != "Unknown":
                    recognized_users[tuple(res['box'])] = name  # Track the recognized user
                    last_recognition_time = current_time  # Update recognition time
                else:
                    # Check if this face was recognized before (within the session)
                    if tuple(res['box']) in recognized_users:
                        name = recognized_users[tuple(res['box'])]  # Use previously recognized name

                # Smooth the bounding box coordinates
                x, y, w, h = res['box']
                smoothed_box = smooth_bounding_box((x, y, w, h))
                x, y, w, h = smoothed_box

                box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for known, Red for unknown
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, box_color, 1)

                # Show accuracy rate at the bottom of the box formatted as a percentage
                accuracy_percentage = max_prob * 100  # Convert to percentage
                cv2.putText(frame, f'Accuracy: {accuracy_percentage:.2f}%', (x, y + h + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, box_color, 1)

                # Current date and time
                now = datetime.now()
                entry_datetime = now.strftime("%m/%d/%Y %I:%M:%S")  # 12-hour format with AM/PM

                if name != "Unknown":
                    current_detection["name"] = name
                    current_detection["confidence"] = max_prob
                    current_user_name = name

                    # Look for the user's corresponding folder
                    grade_level, section = None, None
                    for root, dirs, files in os.walk('datasets'):
                        for dir in dirs:
                            section_path = os.path.join(root, dir)
                            for student_file in os.listdir(section_path):
                                if name in student_file:
                                    grade_level = os.path.basename(os.path.dirname(section_path))
                                    section = dir
                                    break
                            if grade_level and section:
                                break

                    current_detection["grade_level"] = grade_level if grade_level else "Unknown"
                    current_detection["section"] = section if section else "Unknown"
                    current_detection["datetime"] = entry_datetime + " " + now.strftime("%p")  # Include AM/PM
                    
                     # Fetch the user's picture path
                    user_picture_path = get_user_picture(name, grade_level, section)
                    current_detection["picture_path"] = user_picture_path
                                    
                    # Insert data into the database if the interval has passed
                    if name not in last_insertion_times:
                        last_insertion_times[name] = now - insert_interval  # Set initial time

                    if now - last_insertion_times[name] >= insert_interval:
                        async_insert_data(name, entry_datetime, now.strftime("%p"), current_detection["grade_level"], current_detection["section"])  
                        upsert_attendance(name, current_detection["grade_level"], current_detection["section"])
                        last_insertion_times[name] = now  # Update last insertion time

                detected_info.update(current_detection)
                last_valid_detection = current_detection # Update last valid detection
                
                # Fetch current user name
                threading.Thread(target=async_fetch_user_data, args=(name,)).start()

            else:
                detected_info.update(last_valid_detection)

        else:
            detected_info.update(last_valid_detection)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@faceRecognition_bp.route('/current_user', methods=['GET'])
def current_user(): 
    return jsonify({'name': current_user_name})

@faceRecognition_bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@faceRecognition_bp.route('/get_detected_info')
def get_detected_info():
    if not camera_running:  # Check if the camera is running
        return jsonify({"name": "Unknown", "datetime": None, "grade_level": None, "section": None})  # Default response
    return jsonify(detected_info)

    



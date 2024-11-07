from flask import Blueprint, render_template, redirect, url_for, session, jsonify, send_file
import sqlite3
from pathlib import Path
from datetime import datetime
import os
import shutil

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('auth.login'))
    return render_template('views/dashboard.html')

def get_db_connection():
    conn = sqlite3.connect('face-recognition.db')
    conn.row_factory = sqlite3.Row
    return conn

def insert_activity_log(name, activity, date_time):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO activitylogs (name, activity, date_time)
        VALUES (?, ?, ?)
    ''', (name, activity, date_time))

    conn.commit()
    conn.close()

@dashboard_bp.route('/logout')
def logout():
    date_time = datetime.now().strftime('%m/%d/%Y %I:%M:%S %p')

    username = session.get('username', 'Unknown User')  
    insert_activity_log(username, 'Logged out', date_time)

    session.pop('logged_in', None)
    session.pop('username', None) 
    return redirect(url_for('auth.login'))

@dashboard_bp.route('/get_all_users', methods=['GET'])
def get_all_users():
    conn = get_db_connection()
    users = conn.execute('SELECT * FROM users').fetchall()
    conn.close()
    
    users_list = [dict(user) for user in users]
    return jsonify(users_list)

@dashboard_bp.route('/get_registered_users', methods=['GET'])
def get_registered_users():
    conn = get_db_connection()
    attendances = conn.execute('SELECT * FROM attendance').fetchall()
    conn.close()
    
    attendance_list = [dict(attendance) for attendance in attendances]
    return jsonify(attendance_list)

@dashboard_bp.route('/count_all_users', methods=['GET'])
def count_all_users():
    conn = get_db_connection()
    # Count distinct names directly in the query
    count = conn.execute('SELECT COUNT(DISTINCT name) FROM attendance').fetchone()[0]
    conn.close()
    
    return jsonify({'unique_users_count': count})

@dashboard_bp.route('/get_activity_logs', methods=['GET'])
def get_activity_logs():
    try:
        conn = get_db_connection()
        logs = conn.execute('SELECT * FROM activitylogs').fetchall()

        columns = [column[0] for column in conn.execute('SELECT * FROM activitylogs').description]
        
        logs_list = [dict(zip(columns, log)) for log in logs]

        conn.close()
        return jsonify(logs_list)
    
    except Exception as e:
        print(f"Error fetching activity logs: {e}")
        return jsonify({"error": str(e)}), 500
    
@dashboard_bp.route('/backup', methods=['POST'])
def backup_database():
    db_path = 'face-recognition.db'  # Path to the database file in the root directory
    backup_path = 'backup/face_recognition_backup.db'  # Backup path

    # Ensure the backup directory exists
    os.makedirs('backup', exist_ok=True)

    # Copy the database file to create a backup
    shutil.copyfile(db_path, backup_path)

    # Return the backup file for download
    return send_file(backup_path, as_attachment=True)
   


from flask import Blueprint, render_template, redirect, url_for, session, jsonify, send_file
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import os
import shutil
import calendar

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
    users = conn.execute('SELECT * FROM attendance').fetchall()
    conn.close()
    
    users_list = [dict(user) for user in users]
    return jsonify(users_list)

@dashboard_bp.route('/get_registered_users', methods=['GET'])
def get_registered_users():
    conn = get_db_connection()
    attendances = conn.execute('SELECT * FROM users').fetchall()
    conn.close()
    
    attendance_list = [dict(attendance) for attendance in attendances]
    return jsonify(attendance_list)

@dashboard_bp.route('/count_all_users', methods=['GET'])
def count_all_users():
    conn = get_db_connection()

    # Count distinct students
    students_count = conn.execute('SELECT COUNT(DISTINCT name) FROM users WHERE user_type = ?', ('Student',)).fetchone()[0]

    # Count distinct teachers
    teachers_count = conn.execute('SELECT COUNT(DISTINCT name) FROM users WHERE user_type = ?', ('Teacher',)).fetchone()[0]

    # Count distinct staff
    staffs_count = conn.execute('SELECT COUNT(DISTINCT name) FROM users WHERE user_type = ?', ('Staff',)).fetchone()[0]

    # Count total users
    total_users_count = conn.execute('SELECT COUNT(DISTINCT name) FROM users').fetchone()[0]

    conn.close()
    
    # Return all counts as JSON
    return jsonify({
        'students_count': students_count,
        'teachers_count': teachers_count,
        'staffs_count': staffs_count,
        'total_users_count': total_users_count
    })
    
@dashboard_bp.route('/count_attendance', methods=['GET'])
def count_attendance():
    # Get today's date in MM/DD/YYYY format
    today_date = datetime.now().strftime('%m/%d/%Y')

    conn = get_db_connection()

    # Count daily attendance for today (compare only the MM/DD/YYYY part)
    query = """
    SELECT COUNT(*) 
    FROM attendance 
    WHERE SUBSTR(time_in, 1, 10) = ?
    """
    daily_attendance = conn.execute(query, (today_date,)).fetchone()[0]

    # Count total attendance records
    total_attendance = conn.execute('SELECT COUNT(*) FROM attendance').fetchone()[0]

    conn.close()

    # Return the counts as JSON
    return jsonify({
        'total_attendance': total_attendance,
        'daily_attendance': daily_attendance
    })
    
@dashboard_bp.route('/monthly_attendance', methods=['GET'])
def get_monthly_attendance():
    # Manually define the months from November 2024 to October 2025
    months = [
        '11/2024', '12/2024', '01/2025', '02/2025', '03/2025', '04/2025', 
        '05/2025', '06/2025', '07/2025', '08/2025', '09/2025', '10/2025'
    ]

    # Map MM/YYYY format to full month names (e.g., "November 2024")
    month_names = [
        f"{calendar.month_name[int(month.split('/')[0])]} {month.split('/')[1]}"
        for month in months
    ]

    # Log the months for debugging
    #print("Months to query:", months)

    # Query the database to get attendance counts for each month
    conn = get_db_connection()
    query = """
    SELECT SUBSTR(time_in, 1, 2) || '/' || SUBSTR(time_in, 7, 4) AS month, COUNT(*)
    FROM attendance
    GROUP BY month
    ORDER BY month ASC
    """
    result = conn.execute(query).fetchall()
    conn.close()

    # Process the result to match with our manually defined months, using a dictionary for quick lookup
    result_dict = dict(result)  # Map 'MM/YYYY' -> count

    # Get attendance counts for each month, defaulting to 0 if not found
    attendance_counts = [result_dict.get(month, 0) for month in months]

    # Log the extracted labels and counts for debugging
    #print("Labels:", month_names)
    #print("Attendance Counts:", attendance_counts)

    # Return the data as JSON
    return jsonify({
        'labels': month_names,
        'attendance_counts': attendance_counts
    })

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
   

'''
attendance_data = {
    "11/2024": 3,
    "12/2024": 2,
    "01/2025": 2,
    "02/2025": 2,
    "03/2025": 2,
    "04/2025": 10,
    "05/2025": 9,
    "06/2025": 4,
    "07/2025": 0,
    "08/2025": 4,
    "09/2025": 2,
    "10/2025": 1
}

@dashboard_bp.route('/monthly_attendance', methods=['GET'])
def get_monthly_attendance():
    # Manually define the months from November 2024 to October 2025
    months = [
        '11/2024', '12/2024', '01/2025', '02/2025', '03/2025', '04/2025', 
        '05/2025', '06/2025', '07/2025', '08/2025', '09/2025', '10/2025'
    ]

    # Map MM/YYYY format to full month names (e.g., November 2024)
    month_names = []
    for month in months:
        month_num = int(month.split('/')[0])  # Extract the month number (MM)
        month_name = calendar.month_name[month_num]  # Convert to full month name
        year = month.split('/')[1]  # Extract the year (YYYY)
        month_names.append(f"{month_name} {year}")  # Format: "November 2024"

    # Prepare attendance counts based on the mock data
    attendance_counts = [attendance_data.get(month, 0) for month in months]

    # Return the data as JSON
    return jsonify({
        'labels': month_names,
        'attendance_counts': attendance_counts
    })
'''
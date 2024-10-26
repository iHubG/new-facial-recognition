from flask import Blueprint, render_template, request, redirect, url_for, session
import sqlite3
import bcrypt
from pathlib import Path
from datetime import datetime

auth_bp = Blueprint('auth', __name__)

def get_db_connection():
    conn = sqlite3.connect('face-recognition.db')
    conn.row_factory = sqlite3.Row
    return conn 

def verify_password(stored_hash, password):
    return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))

def insert_activity_log(name, activity, date_time):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO activitylogs (name, activity, date_time)
        VALUES (?, ?, ?)
    ''', (name, activity, date_time))

    conn.commit()
    conn.close()

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    errors = {}
    success_message = None
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username:
            errors['username'] = 'Username is required'
        if not password:
            errors['password'] = 'Password is required'
        elif len(password) < 8:
            errors['password'] = 'Password must be at least 8 characters long'

        if not errors:
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute('SELECT password_hash FROM admin WHERE username = ?', (username,))
                result = cursor.fetchone()
                conn.close()

                if result:
                    stored_hash = result[0]
                    if verify_password(stored_hash, password):
                        session['logged_in'] = True
                        session['username'] = username
                        
                        date_time = datetime.now().strftime('%m/%d/%Y %I:%M:%S %p')

                        insert_activity_log(username, 'Logged in', date_time)
                        
                        success_message = 'Login successful! Redirecting...' 
                    else:
                        errors['auth'] = 'Username and Password do not match!'
                else:
                    errors['auth'] = 'Username not found!'

            except sqlite3.Error as e:
                errors['auth'] = f"An error occurred: {e}"
    
    return render_template('views/login.html', errors=errors, success_message=success_message)

from flask import Blueprint, render_template, request, redirect, url_for, session
import sqlite3
import bcrypt
from pathlib import Path

auth_bp = Blueprint('auth', __name__)

def get_db_connection():
    db_path =  db_path = Path(__file__).resolve().parent.parent / 'model' / 'face.db'
    return sqlite3.connect(db_path, timeout=10)  

def verify_password(stored_hash, password):
    return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))

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
                cursor.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
                result = cursor.fetchone()
                conn.close()

                if result:
                    stored_hash = result[0]
                    if verify_password(stored_hash, password):
                        session['logged_in'] = True
                        session['username'] = username
                        success_message = 'Login successful! Redirecting...' 
                    else:
                        errors['auth'] = 'Username and Password do not match!'
                else:
                    errors['auth'] = 'Username not found!'

            except sqlite3.Error as e:
                errors['auth'] = f"An error occurred: {e}"
    
    return render_template('views/login.html', errors=errors, success_message=success_message)

from flask import Blueprint, render_template, redirect, url_for, session, jsonify
import sqlite3

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('auth.login'))
    return render_template('views/dashboard.html')

@dashboard_bp.route('logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('auth.login'))

def get_db_connection():
    conn = sqlite3.connect('face-recognition.db')
    conn.row_factory = sqlite3.Row
    return conn

@dashboard_bp.route('/get_all_users', methods=['GET'])
def get_all_users():
    conn = get_db_connection()
    users = conn.execute('SELECT * FROM users').fetchall()
    conn.close()
    
    users_list = [dict(user) for user in users]
    return jsonify(users_list)

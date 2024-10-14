from flask import Blueprint, render_template, redirect, url_for, session

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

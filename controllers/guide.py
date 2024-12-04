from flask import Blueprint, render_template, redirect, url_for, session, jsonify, send_file

guide_bp = Blueprint('instruction', __name__)

@guide_bp.route('instruction')
def guide():

    return render_template('views/guide.html')
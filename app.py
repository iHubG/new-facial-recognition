from flask import Flask, render_template, redirect, url_for
from controllers.auth import auth_bp
from controllers.dashboard import dashboard_bp
from controllers.faceRecognition import faceRecognition_bp, camera
import ssl
import os
import secrets

app = Flask(__name__)

def load_or_generate_secret_key():
    key_file = 'secret_key.txt'

    if os.path.exists(key_file):
        with open(key_file, 'r') as f:
            key = f.read().strip()
            return key  
    else:
        secret_key = secrets.token_hex(16)  
        with open(key_file, 'w') as f:
            f.write(secret_key) 
        return secret_key 


app.secret_key = load_or_generate_secret_key()

app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(dashboard_bp, url_prefix='/admin')
app.register_blueprint(faceRecognition_bp, url_prefix='/user')

@app.route('/')
def index():
    camera.stop()
    return redirect(url_for('auth.login'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('views/404.html'), 404

if __name__ == '__main__':
    cert_path = os.path.join('certs', '127.0.0.1.pem')
    key_path = os.path.join('certs', '127.0.0.1-key.pem')

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=cert_path, keyfile=key_path)

    app.run(host='127.0.0.1', port=5500, ssl_context=context, debug=True)

#for rule in app.url_map.iter_rules():
#    print(rule)



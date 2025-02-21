from flask import Flask, render_template, request, jsonify, redirect, session, url_for
from flask_socketio import SocketIO
import firebase_admin
from firebase_admin import credentials, auth
import pyrebase
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.urandom(24)  # Generate a random secret key
socketio = SocketIO(app)

try:
    # Firebase Configuration
    cred = credentials.Certificate("examguard/serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

    firebase_config = {
        "apiKey": "AIzaSyAK8aMoxmT7hpbxl8akPYitfxFM6Ytl76I",
        "authDomain": "examguard-login.firebaseapp.com",
        "databaseURL": "https://firestore.googleapis.com/v1/projects/examguard-login/databases/(default)/documents/",
        "projectId": "examguard-login",
        "storageBucket": "examguard-login.firebasestorage.app",
        "messagingSenderId": "229046359229",
        "appId": "1:229046359229:web:5f67beec4538e13b36d3d4"
    }

    firebase = pyrebase.initialize_app(firebase_config)
    auth_firebase = firebase.auth()
    logger.info("Firebase initialized successfully")

except Exception as e:
    logger.error(f"Failed to initialize Firebase: {str(e)}")
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if "user" not in session:
        logger.warning("Unauthorized access attempt to dashboard")
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session["user"])

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if "user" in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        try:
            email = request.form.get('email')
            password = request.form.get('password')
            first_name = request.form.get('firstName')
            last_name = request.form.get('lastName')
            
            # Create user in Firebase
            user = auth_firebase.create_user_with_email_and_password(email, password)
            
            # Update user profile with name
            auth_firebase.update_profile(user['idToken'], 
                                      display_name=f"{first_name} {last_name}")
            
            logger.info(f"User created successfully: {email}")
            return redirect(url_for('login'))
            
        except Exception as e:
            logger.error(f"Signup error: {str(e)}")
            return render_template('signup.html', error="Failed to create account. Please try again.")
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if "user" in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        try:
            email = request.form.get('email')
            password = request.form.get('password')
            
            # Authenticate with Firebase
            user = auth_firebase.sign_in_with_email_and_password(email, password)
            
            # Create session
            session["user"] = email
            session["uid"] = user['localId']
            
            logger.info(f"User logged in successfully: {email}")
            return redirect(url_for('dashboard'))
            
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return render_template('login.html', error="Invalid email or password")
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    user_email = session.get("user", "unknown")
    session.clear()
    logger.info(f"User {user_email} logged out")
    return redirect(url_for('login'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Unhandled error: {str(error)}")
    return jsonify({
        "status": "error",
        "message": "An unexpected error occurred"
    }), 500

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def emit_detection_update(detection_data):
    socketio.emit('detection_update', detection_data)

if __name__ == '__main__':
    app.debug = True
    logger.info("Starting application")
    socketio.run(app, host='0.0.0.0', port=5000)
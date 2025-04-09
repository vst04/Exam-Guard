import pyrebase

config = {
    "apiKey": "AIzaSyAK8aMoxmT7hpbxl8akPYitfxFM6Ytl76I",
    "authDomain": "examguard-login.firebaseapp.com",
    "projectId": "examguard-login",
    "storageBucket": "examguard-login.firebasestorage.app",
    "messagingSenderId": "229046359229",
    "appId": "1:229046359229:web:5f67beec4538e13b36d3d4",
    "databaseURL": "https://firestore.googleapis.com/v1/projects/examguard-login/databases/(default)/documents/"
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()

# Use the existing email
email = "newtest@example.com"
password = "testpassword123"

try:
    user = auth.sign_in_with_email_and_password(email, password)
    print("✅ Successfully logged in as:", user["email"])
except Exception as e:
    print("❌ Login failed:", e)

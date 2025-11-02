from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import cv2
import sqlite3
import os
import requests
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==============================
# Load trained model from external storage
# ==============================
MODEL_URL = os.environ.get('MODEL_URL') # Set in Render environment variables
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "face_emotionModel.h5")
 # Render free-tier temp folder

# Download model if it doesn't exist locally
if not os.path.exists(MODEL_PATH):
    print("Downloading model from external storage...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("Model downloaded!")

# Load the model
model = load_model(MODEL_PATH)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ==============================
# Database setup
# ==============================
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            department TEXT,
            image_path TEXT,
            emotion TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ==============================
# Function to predict emotion
# ==============================
def predict_emotion(img_path, debug=False):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48))
    gray = gray.astype('float32') / 255.0
    gray = np.expand_dims(gray, axis=(0, -1))

    preds = model.predict(gray)
    probs = preds[0]
    emotion_index = np.argmax(probs)
    emotion = emotion_labels[emotion_index]

    if debug:
        print(f"Predicted probabilities: {probs}")
        print(f"Detected emotion: {emotion}")

    return emotion

# ==============================
# Routes
# ==============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    name = request.form['name']
    email = request.form['email']
    department = request.form['department']
    file = request.files['photo']

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Predict emotion
        emotion = predict_emotion(file_path)

        # Save to database
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute(
            'INSERT INTO students (name, email, department, image_path, emotion) VALUES (?, ?, ?, ?, ?)',
            (name, email, department, file_path, emotion)
        )
        conn.commit()
        conn.close()

        # Friendly response messages
        responses = {
            'Happy': "You are smiling. You look happy today 😊",
            'Sad': "You look sad. Hope everything is okay 💙",
            'Angry': "You seem upset. Take a deep breath 😤",
            'Fear': "You look scared. Don’t worry, you got this 😨",
            'Disgust': "You look disgusted. Something bothering you? 🤢",
            'Neutral': "You look calm and neutral 😐",
            'Surprise': "Wow! You look surprised 😲"
        }

        message = responses.get(emotion, "Emotion detected.")

        return f"<h2>{message}</h2><p>Detected emotion: {emotion}</p><a href='/'>Try again</a>"

    return "No image uploaded. Please try again."

# ==============================
# Run app
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


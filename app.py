from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import sqlite3
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
model = load_model('face_emotionModel.h5')

# Emotion labels (adjust if your model has different order)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# Database setup
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

# Function to predict emotion
def predict_emotion(img_path, debug=False):
    # Load the uploaded image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")

    # Convert to grayscale since model expects 1 channel
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to 48x48 (model input size)
    gray = cv2.resize(gray, (48, 48))

    # Normalize (same as training)
    gray = gray.astype('float32') / 255.0

    # Reshape to (1, 48, 48, 1)
    gray = np.expand_dims(gray, axis=(0, -1))

    # Predict
    preds = model.predict(gray)
    probs = preds[0]
    emotion_index = np.argmax(probs)
    emotion = emotion_labels[emotion_index]

    if debug:
        print(f"Predicted probabilities: {probs}")
        print(f"Detected emotion: {emotion}")

    return emotion


# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission
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
        c.execute('INSERT INTO students (name, email, department, image_path, emotion) VALUES (?, ?, ?, ?, ?)',
                  (name, email, department, file_path, emotion))
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

if __name__ == '__main__':
    app.run(debug=True)


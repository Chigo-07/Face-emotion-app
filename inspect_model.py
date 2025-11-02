# inspect_model.py
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("face_emotionModel.h5")
print("MODEL SUMMARY")
model.summary()

print("\nInput shape (model.input_shape):", model.input_shape)
print("Output shape (model.output_shape):", model.output_shape)

# If model saved class indices mapping externally, print hint:
# We can't read training-time class order from the H5 unless you saved metadata.
# So we'll assume standard FER order: [Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral]


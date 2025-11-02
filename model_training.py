# model_training.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# 1️⃣ Define dataset paths
train_dir = os.path.join("fer2013", "train")
test_dir = os.path.join("fer2013", "test")

# 2️⃣ Image preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,           # normalize pixel values
    rotation_range=30,        # rotate images randomly
    width_shift_range=0.2,    # horizontal shift
    height_shift_range=0.2,   # vertical shift
    shear_range=0.2,          # shear
    zoom_range=0.2,           # zoom
    horizontal_flip=True,     # flip images
    fill_mode='nearest'       # fill in missing pixels
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 3️⃣ Load images directly from the folders
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),     # resize images to 48x48
    batch_size=64,
    color_mode='grayscale',   # FER2013 images are grayscale
    class_mode='categorical'  # output as one-hot encoded labels
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

# 4️⃣ Build the CNN model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dense(7, activation='softmax')   # 7 emotion classes
])

# 5️⃣ Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6️⃣ Train the model
epochs = 25  # you can increase later if you want better accuracy
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs
)

# 7️⃣ Save the trained model
model.save("face_emotionModel.h5")

print("✅ Model training complete. Saved as 'face_emotionModel.h5'")


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt

# Define parameters
BATCH_SIZE = 32
IMAGE_SIZE = (128, 128)
EPOCHS = 20
CLASSES = ['elephant', 'giraffe', 'lion', 'tiger',
           'bear', 'red panda', 'kangaroo', 'panda',
           'crocodile', 'penguin', 'jaguar (animal)',
           'rhinoceros', 'hippopotamus', 'monkey']

# 1. Load and Prepare the Dataset
def load_data(data_dir):
    images = []
    labels = []
    for idx, animal_class in enumerate(CLASSES):
        # Now we look in the 'images' subfolder of each class
        class_dir = os.path.join(data_dir, animal_class, 'images')
        if not os.path.exists(class_dir):
            print(f"Directory {class_dir} not found!")
            continue
        # Find all .jpg images within the class's 'images' subfolder
        image_paths = glob(os.path.join(class_dir, '*.jpg'))
        if len(image_paths) == 0:
            print(f"No images found for class {animal_class} in directory {class_dir}.")
        for image_path in image_paths:
            # Load image, resize, and convert to array
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
            img = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img)
            labels.append(idx)  # Assign a class index for each animal type
    images = np.array(images, dtype='float32') / 255.0  # Normalize to [0,1]
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(CLASSES))
    return images, labels

# 2. Load data and split into train, validation, and test sets
data_dir = 'openimages_zoo_animals15'
images, labels = load_data(data_dir)
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# 3. Define the CNN Model Architecture
def build_model(input_shape=(128, 128, 3), num_classes=len(CLASSES)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# 4. Compile the Model
model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)

val_datagen = ImageDataGenerator()  # No augmentation for validation
val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

# 6. Train the Model
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=len(X_val) // BATCH_SIZE
)

# Save accuracy plot
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png')  # Save the plot as a PNG file
plt.close()

# Save loss plot
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')  # Save the plot as a PNG file
plt.close()

# 7. Evaluate the Model
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation accuracy: {val_accuracy:.2f}")

# Save the trained model
model.save('zoo_animal_classifier15-20epochs-100each.keras')
model.save('zoo_animal_classifier15-20epochs-100each.h5')
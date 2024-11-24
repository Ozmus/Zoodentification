import tensorflow as tf
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

# Define the classes (must match the classes used during training)
CLASSES = ['elephant', 'giraffe', 'zebra', 'lion', 'tiger']

# Function to preprocess an input image
def preprocess_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)  # Load and resize the image
    img_array = img_to_array(img)  # Convert to a NumPy array
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Load the trained model
model = tf.keras.models.load_model('zoo_animal_classifier.h5')

# Path to the image you want to classify
image_path = 'test_images/lion/lion.jpg'

# Preprocess the image
image = preprocess_image(image_path, target_size=(128, 128))

# Make a prediction
predictions = model.predict(image)

# Get the index of the highest probability
predicted_class_index = np.argmax(predictions)

# Get the class label
predicted_class = CLASSES[predicted_class_index]

confidence = predictions[0][predicted_class_index]
print(f"Predicted class: {predicted_class} with confidence: {confidence:.2f}")
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

# Define the classes (must match the classes used during training)
CLASSES = ['elephant', 'giraffe', 'lion', 'tiger',
           'bear', 'red panda', 'kangaroo', 'panda',
           'crocodile', 'penguin', 'jaguar (animal)',
           'rhinoceros', 'hippopotamus', 'monkey']

# Function to preprocess an input image
def preprocess_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)  # Load and resize the image
    img_array = img_to_array(img)  # Convert to a NumPy array
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Load the trained model
model = tf.keras.models.load_model('zoo_animal_classifier15-20epochs-100each.h5')

# Path to the image you want to classify
image_path = 'test-images/{{animal_class}}/{{animal_class}}{{number}}.jpg'
true_count = 0
false_count = 0
for animal_class in CLASSES:
    if animal_class != 'red panda':
        for i in range(1, 7):
            input_image_path = image_path.replace('{{animal_class}}', animal_class).replace('{{number}}', str(i))
            print(input_image_path)
            # Preprocess the image
            image = preprocess_image(input_image_path, target_size=(128, 128))

            # Make a prediction
            predictions = model.predict(image)

            # Get the index of the highest probability
            predicted_class_index = np.argmax(predictions)

            # Get the class label
            predicted_class = CLASSES[predicted_class_index]

            confidence = predictions[0][predicted_class_index]
            print(f"Predicted class: {predicted_class} with confidence: {confidence:.2f}")
            if predicted_class == animal_class:
                true_count += 1
            else:
                false_count += 1

print(true_count)
print(false_count)
'''
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
'''
'''from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = load_model('/Users/srivasr1/Desktop/Project/best_model_weights.h5')

def preprocess_image(image, target_size):
    if image.mode != "L":  # Check if image is not grayscale
        image = image.convert("L")  # Convert to grayscale
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    # Check if the file was sent
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    # Load the image from the request
    file = request.files['file']

    # Preprocess the image
    image = Image.open(file)
    processed_image = preprocess_image(image, target_size=(128, 128))

    # Predict
    prediction = model.predict([processed_image]).tolist()

    # Create the response
    response = {
        'prediction': {
            'parkinson': prediction[0][0],
            'healthy': 1 - prediction[0][0]  # Assuming your model outputs the probability of parkinson
        }
    }
    return jsonify(response)
'''
import base64
from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
from flask_cors import CORS
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)
CORS(app)
model = load_model('/Users/srivasr1/Desktop/Project/best_model_weights.h5')

# Define image data generator for augmentation
data_generator = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.3, 1.15],
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    preprocessing_function=None,
    validation_split=0.0,
    dtype='float32'
)

def preprocess_image(image, target_size):
    if image.mode != "L":  # Check if image is not grayscale
        image = image.convert("L")  # Convert to grayscale
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    # Check if file was sent
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    # Load the image from the request
    file = request.files['file']
    
    # Preprocess the original image
    original_image = Image.open(file)
    processed_image = preprocess_image(original_image, target_size=(128, 128))
    
    # Augment the image and create 25 images
    augmented_images = []
    for _ in range(25):
        augmented_image = data_generator.random_transform(processed_image[0])
        augmented_images.append(augmented_image)

    # Initialize an empty list to store predictions for each augmented image
    all_predictions = []
    
    # Loop over augmented images and obtain predictions
    for augmented_image in augmented_images:
        # Obtain prediction for the augmented image
        prediction = model.predict(np.expand_dims(augmented_image, axis=0)).tolist()
        # Append prediction to the list
        all_predictions.append(prediction)

    # Convert the list of predictions to a NumPy array for easier calculation
    all_predictions = np.array(all_predictions)

    # Calculate the average prediction by taking the mean across all predictions
    average_prediction = np.mean(all_predictions, axis=0)
    
    # Console log the average prediction
    print("Average Prediction:", average_prediction)
    
    # Create the response
    response = {
        'original_prediction': {
            'parkinson': prediction[0][0],
            'healthy': 1 - prediction[0][0]  # Assuming your model outputs the probability of parkinson
        },
        'augmented_predictions': all_predictions.tolist(),
        'average_prediction': average_prediction.tolist()
    }
    return jsonify(response)

if __name__ == "_main_":
    app.run(debug=True)
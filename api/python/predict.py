# This file is located at /api/python/predict.py

from flask import Flask, request, jsonify
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
import tensorflow as tf
import os
import traceback

# Vercel will create an 'app' variable from this file
app = Flask(__name__)

# --- Suppress TensorFlow informational messages ---
# This should be done before any major TensorFlow operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Load the Keras Model ---
# We load the model once when the serverless function starts up.
# This is efficient because Vercel keeps the function "warm" for subsequent requests.
# The path must be relative to this script's location.
try:
    model_path = os.path.join(os.path.dirname(__file__), 'champion_model.keras')
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    # If the model fails to load, we create a placeholder so the app doesn't crash on startup.
    # The error will be reported when a prediction is attempted.
    model = None
    model_load_error = traceback.format_exc()

# --- Haralick Feature Calculation Helper Function ---
# This is the same function we used in our previous Python scripts.
def calculate_haralick_features(image):
    glcm = graycomatrix(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    energy = graycoprops(glcm, 'energy').mean()
    return np.array([contrast, homogeneity, correlation, energy])

# --- Define the API Route ---
# Vercel routes all requests to this file to the 'app' object.
# The methods=['POST'] ensures this function only responds to POST requests.
@app.route('/', defaults={'path': ''}, methods=['POST'])
@app.route('/<path:path>', methods=['POST'])
def predict(path):
    # First, check if the model loaded correctly during startup.
    if model is None:
        return jsonify({
            "error": "Model failed to load on the server.",
            "details": model_load_error
        }), 500

    try:
        # Check if an image file was included in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided in the request.'}), 400

        file = request.files['image']
        
        # Read the image file data into memory as a byte stream
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        
        # Decode the byte stream into an OpenCV image (in BGR format)
        img_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({'error': 'Could not decode the image file. It might be corrupted.'}), 400

        # --- PREPARE INPUTS FOR THE MODEL ---
        # 1. Image Input
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_normalized = img_resized / 255.0
        image_input = np.expand_dims(img_normalized, axis=0) # Add batch dimension

        # 2. Haralick Features Input
        img_gray = cv2.cvtColor(img_resized, cv.COLOR_RGB2GRAY)
        haralick_features = calculate_haralick_features(img_gray)
        haralick_input = np.expand_dims(haralick_features, axis=0) # Add batch dimension
        
        # --- RUN PREDICTION ---
        prediction = model.predict([image_input, haralick_input], verbose=0)
        
        # --- FORMAT AND RETURN THE RESPONSE ---
        class_labels = ['Healthy', 'FMD Diseased']
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels[predicted_class_index]
        confidence = float(np.max(prediction))
        
        result = {
            "prediction": predicted_class_label,
            "confidence": f"{confidence * 100:.2f}%"
        }
        
        return jsonify(result)

    except Exception as e:
        # If any other error happens during prediction, return a detailed error message.
        return jsonify({
            "error": "An error occurred during prediction.",
            "details": str(e),
            "trace": traceback.format_exc()
        }), 500
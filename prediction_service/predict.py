# In prediction_service/predict.py

import sys
import json
import numpy as np
import cv2
import traceback

# --- This block must be BEFORE the tensorflow import ---
# It suppresses TensorFlow's informational messages (like "Your CPU supports...")
# 0 = all messages are logged (default)
# 1 = INFO messages are filtered out
# 2 = INFO and WARNING messages are filtered out
# 3 = INFO, WARNING, and ERROR messages are filtered out
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from skimage.feature import graycomatrix, graycoprops
# --- End of suppression block ---


# This function does not need to change
def calculate_haralick_features(image):
    glcm = graycomatrix(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    energy = graycoprops(glcm, 'energy').mean()
    return np.array([contrast, homogeneity, correlation, energy])

def main(image_path):
    try:
        # --- Load Model and Image ---
        # The model is now loaded inside the main function
        model = tf.keras.models.load_model('champion_model.keras')
        
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            # This is a specific error we can handle
            raise FileNotFoundError(f"Python script could not find or read image at {image_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # --- PREPARE INPUTS ---
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_normalized = img_resized / 255.0
        image_input = np.expand_dims(img_normalized, axis=0)
        
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        haralick_features = calculate_haralick_features(img_gray)
        haralick_input = np.expand_dims(haralick_features, axis=0)

        # --- RUN PREDICTION ---
        prediction = model.predict([image_input, haralick_input], verbose=0)
        
        # --- FORMAT RESULT ---
        class_labels = ['Healthy', 'FMD Diseased']
        predicted_class_index = np.argmax(prediction)
        
        # Prepare a dictionary for the JSON output
        result = {
            "prediction": class_labels[predicted_class_index],
            "confidence": f"{float(np.max(prediction)) * 100:.2f}%"
        }
        # Return the result as a JSON string
        return json.dumps(result)

    except Exception as e:
        # If ANY error occurs, create a JSON object with the error message
        error_result = {
            "error": "An error occurred in the Python script.",
            "details": str(e),
            "trace": traceback.format_exc() # Provides the full error traceback
        }
        # Return the error as a JSON string
        return json.dumps(error_result)

if __name__ == '__main__':
    # The image path will be passed as a command-line argument
    if len(sys.argv) > 1:
        # Call the main function and print whatever it returns (either success or error JSON)
        output_json = main(sys.argv[1])
        print(output_json)
    else:
        # Handle the case where no image path is provided
        print(json.dumps({"error": "No image path provided to Python script"}))
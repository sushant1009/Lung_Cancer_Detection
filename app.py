from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import os
import logging
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import gc

# Initialize Flask App
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Define Upload Folder
UPLOAD_FOLDER = "/tmp/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Logging Setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load TFLite Model (quantized for predictions)
TFLITE_MODEL_PATH = os.path.join(os.path.dirname(__file__), '1_deploy_final_cancer_model.tflite')
H5_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'final_cancer_model.h5')

if not os.path.exists(TFLITE_MODEL_PATH):
    logger.error(f"TFLite model file {TFLITE_MODEL_PATH} not found.")
    raise FileNotFoundError(f"TFLite model file {TFLITE_MODEL_PATH} not found.")
if not os.path.exists(H5_MODEL_PATH):
    logger.error(f"Keras model file {H5_MODEL_PATH} not found.")
    raise FileNotFoundError(f"Keras model file {H5_MODEL_PATH} not found.")

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = (224, 224)
class_names = ['Benign', 'Malignant', 'Normal']  # Verify this matches training order
logger.info("TFLite (quantized) model loaded successfully")

# Preprocessing Function (aligned with training)
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(IMG_SIZE)
        img = np.array(img, dtype=np.float32)
        # Uncomment the next line if your model was trained with EfficientNet preprocessing
        # img = tf.keras.applications.efficientnet.preprocess_input(img)
        img = img / 255.0  # Default normalization (keep this if unsure about training)
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        raise

# Generate Heatmap (load Keras model on-demand)
def generate_heatmap(img_array, class_idx):
    keras_model = tf.keras.models.load_model(H5_MODEL_PATH)
    try:
        # Use a specific deep layer for better heatmap (EfficientNetB0 example)
        last_conv_layer = keras_model.get_layer('block7a_project_conv')  # Adjust if needed
        grad_model = tf.keras.models.Model([keras_model.inputs], [last_conv_layer.output, keras_model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]
        grads = tape.gradient(loss, conv_outputs)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        # Improved normalization
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        heatmap = cv2.resize(heatmap, IMG_SIZE)
        return heatmap
    except Exception as e:
        logger.error(f"Error in heatmap generation: {str(e)}")
        return None
    finally:
        del keras_model  # Free memory
        gc.collect()

# Overlay Heatmap on Original Image
def overlay_heatmap(original_img, heatmap):
    try:
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        original_img = np.uint8(original_img[0] * 255)
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        img_pil = Image.fromarray(superimposed_img)
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        logger.error(f"Error overlaying heatmap: {str(e)}")
        return None

# Serve the Main Webpage
@app.route("/")
def home():
    return render_template("index.html")

# Route for Demo Page
@app.route("/demo")
def demo():
    return render_template("demo.html")

# Serve Static Files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory("static", filename)

# Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    
    try:
        file.save(file_path)
        logger.info(f"Image saved at {file_path}")

        img = preprocess_image(file_path)

        # TFLite prediction
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        probabilities = prediction
        class_probs = {
            'Benign': float(probabilities[0]),
            'Malignant': float(probabilities[1]),
            'Normal': float(probabilities[2])
        }

        predicted_class = max(class_probs, key=class_probs.get)
        confidence = max(class_probs.values())
        if confidence == 0:
            logger.warning("Confidence score is 0, possible model issue.")
            confidence = 1e-6

        class_idx = list(class_probs.keys()).index(predicted_class)
        # Use same preprocessing for heatmap (adjust if EfficientNet-specific needed)
        img_for_heatmap = img  # Or tf.keras.applications.efficientnet.preprocess_input(np.copy(img) * 255.0)
        heatmap = generate_heatmap(img_for_heatmap, class_idx)
        heatmap_base64 = overlay_heatmap(img, heatmap) if heatmap is not None else None

        result = {
            'predicted_class': predicted_class,
            'probability': confidence * 100,
            'confidence': round(confidence * 100, 2),
            'heatmap': heatmap_base64,
            'details': {k: round(v * 100, 2) for k, v in class_probs.items()}
        }

        # Memory cleanup
        del img, prediction, heatmap
        gc.collect()
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted temporary file: {file_path}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
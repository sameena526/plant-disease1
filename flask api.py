# flask_api.py
# A small inference server. Expects a POST /predict with a file upload (image).
# Returns predicted class and probabilities.

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.environ.get('MODEL_PATH', 'models/final_modelold.h5')
CLASS_JSON = os.environ.get('CLASS_JSON', 'models/class_names.json')
IMG_SIZE = int(os.environ.get('IMG_SIZE', 224))

print("Loading model:", MODEL_PATH)
#model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_JSON, 'r') as f:
    class_names = json.load(f)
from tensorflow.keras.utils import custom_object_scope

class TrueDivide(tf.keras.layers.Layer):
    def call(self, inputs):
        # This may not be used, but a placeholder is required for Keras to load
        return inputs

with custom_object_scope({'TrueDivide': TrueDivide}):
    model = tf.keras.models.load_model(MODEL_PATH)

# def prepare_image(image_bytes):
#     img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
#     img = img.resize((IMG_SIZE, IMG_SIZE))
#     arr = np.array(img).astype('float32')
#     # depending on model preprocessing
#     arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
#     arr = np.expand_dims(arr, 0)
#     return arr
def prepare_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        print("❌ PIL could not open the image:", e)
        print("📌 Byte length:", len(image_bytes))
        raise e

    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype('float32')
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, 0)
    return arr

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file part'}), 400
    file = request.files['file']
    img_bytes = file.read()
    x = prepare_image(img_bytes)
    preds = model.predict(x)[0]
    top_idx = int(preds.argmax())
    response = {
        'predicted_class': class_names[top_idx],
        'class_index': int(top_idx),
        'probabilities': {class_names[i]: float(preds[i]) for i in range(len(preds))}
    }
    return jsonify(response)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status':'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)
model = tf.keras.models.load_model("mnist_cnn_model.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    try:
        # Decode image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert('L')
        image = image.resize((28, 28))
        image = np.array(image).astype('float32') / 255.0
        image = image.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(image)
        predicted_digit = int(np.argmax(prediction))

        return jsonify({'prediction': predicted_digit})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

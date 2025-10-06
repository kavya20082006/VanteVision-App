import os
import tensorflow as tf
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

# Correct Flask initialization
app = Flask(__name__)

# Upload folder setup
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the TensorFlow model
MODEL_PATH = "arbitrary-image-stylization-v1-tensorflow-2"
model = tf.saved_model.load(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'content_photo' not in request.files or 'style_photo' not in request.files:
        return 'No file part', 400

    content_file = request.files['content_photo']
    style_file = request.files['style_photo']

    if content_file.filename == '' or style_file.filename == '':
        return 'No selected file', 400

    # Save content image
    content_filename = secure_filename(content_file.filename)
    content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
    content_file.save(content_path)

    # Save style image
    style_filename = secure_filename(style_file.filename)
    style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
    style_file.save(style_path)

    # Preprocess both images
    content_image = load_image(content_path)
    style_image = load_image(style_path)

    # Run style transfer
    try:
        outputs = model(tf.constant(content_image), tf.constant(style_image))
        stylized_image = outputs[0]
    except Exception as e:
        return f"Error during style transfer: {e}"

    # Convert and save stylized image
    stylized_filename = f"stylized_{content_filename}"
    stylized_image_path = os.path.join(app.config['UPLOAD_FOLDER'], stylized_filename)
    tensor_to_image(stylized_image).save(stylized_image_path)

    # Render result page with image
    return render_template('result.html', original_image_name=content_filename,      stylized_image_name=stylized_filename)

# Helper: Load and preprocess image
def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [256, 256])
    img = img[tf.newaxis, :]
    return img

# Helper: Convert tensor to PIL image
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

# Run the Flask server
if __name__ == '__main__':
    print("🚀 Starting Flask server on port 5050...")
    app.run(debug=True, port=5050)
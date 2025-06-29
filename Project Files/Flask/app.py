import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the trained model
model = load_model("poultry_disease_model.h5")  # Make sure the model file is in the same folder

# Correct class label mapping based on alphabetical folder names
class_names = {
    0: "Coccidiosis",
    1: "Healthy",
    2: "New Castle Disease",
    3: "Salmonella"
}

# Image preprocessing function
def prepare_image(filepath):
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        filepath = os.path.join('static', file.filename)
        file.save(filepath)

        img = prepare_image(filepath)
        pred = model.predict(img)
        predicted_class = np.argmax(pred[0])
        confidence = np.max(pred[0])

        label = class_names.get(predicted_class, "Unknown")

        return render_template('index.html',
                               prediction=f"{label} ({confidence*100:.2f}% confidence)",
                               image_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

# Load the Hugging Face model
repo_name = "Jayanth2002/dinov2-base-finetuned-SkinDisease"
image_processor = AutoImageProcessor.from_pretrained(repo_name)
model = AutoModelForImageClassification.from_pretrained(repo_name)

# Define class names
class_names = ['Basal Cell Carcinoma', 'Darier_s Disease', 'Epidermolysis Bullosa Pruriginosa', 
               'Hailey-Hailey Disease', 'Herpes Simplex', 'Impetigo', 'Larva Migrans', 
               'Leprosy Borderline', 'Leprosy Lepromatous', 'Leprosy Tuberculoid', 'Lichen Planus', 
               'Lupus Erythematosus Chronicus Discoides', 'Melanoma', 'Molluscum Contagiosum', 
               'Mycosis Fungoides', 'Neurofibromatosis', 'Papilomatosis Confluentes And Reticulate', 
               'Pediculosis Capitis', 'Pityriasis Rosea', 'Porokeratosis Actinic', 'Psoriasis', 
               'Tinea Corporis', 'Tinea Nigra', 'Tungiasis', 'actinic keratosis', 'dermatofibroma', 
               'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 
               'vascular lesion']

@app.route("/")
def home():
    return jsonify({"message": "Skin Disease Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read()))

        # Preprocess the image
        encoding = image_processor(image.convert("RGB"), return_tensors="pt")

        # Make a prediction
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits

        predicted_class_idx = logits.argmax(-1).item()
        predicted_class_name = class_names[predicted_class_idx]

        return jsonify({"prediction": predicted_class_name})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

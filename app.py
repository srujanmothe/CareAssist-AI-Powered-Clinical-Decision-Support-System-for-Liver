import os
import json
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Predefined classes for fibrosis stages
CLASSES = ["F0", "F1", "F2", "F3", "F4"]

# Load stage descriptions from JSON
with open('stages.json', 'r') as file:
    STAGE_DESCRIPTIONS = json.load(file)

# Define the model architecture and load weights
MODEL_PATH = 'resnet50_fibrosis_model.pth'
model = resnet50(pretrained=False)  # ResNet50 architecture
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(CLASSES))  # Match number of output classes
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))  # Load weights
model.eval()  # Set to evaluation mode

# Define the preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50 expects 224x224 input
    transforms.ToTensor(),         # Convert image to Tensor
    transforms.Normalize(          # Normalize to match ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_stage(image_path):
    """Preprocess the image and predict the fibrosis stage."""
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return CLASSES[predicted.item()]

@app.route('/')
def home():
    """Serve the frontend HTML file."""
    return send_from_directory('.', 'index.html')  # Serve `index.html` from the current directory

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Save the uploaded file
    file = request.files['image']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # Predict the stage
        stage = predict_stage(file_path)
        description = STAGE_DESCRIPTIONS["stages"][stage]
        return jsonify({
            'stage': stage,
            'description': {
                "Major Causes": description["Major Causes"],
                "About the Stage": description["About the Stage"],
                "Accuracy": description["Accuracy"],
                "Preventive Measures": description["Preventive Measures"],
                "Reversibility Chances": description["Reversibility Chances"]
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

# CareAssist: AI-Powered Clinical Decision Support System

## Overview
CareAssist is an **AI-driven Clinical Decision Support System (CDSS)** designed to diagnose liver health across all stages of fibrosis (F0–F4) using ultrasound images. The system leverages **deep learning (CNN + ResNet)** for feature extraction and **machine learning classifiers (Random Forest/XGBoost)** for accurate staging, coupled with **preventive recommendations** for patients.

By integrating evidence-based AI predictions into the healthcare workflow, CareAssist improves diagnosis accuracy, reduces human error, and enables **early detection**—supporting better patient outcomes.

---

## Features
- **Liver Stage Detection**: Classifies liver fibrosis into METAVIR stages F0 to F4 from ultrasound images.
- **Evidence-based Recommendations**: Provides stage-specific precautions and lifestyle advice.
- **Early Prevention**: Detects fibrosis early to avoid progression to cirrhosis.
- **User-Friendly Interface**: Simple upload and result visualization for healthcare professionals.
- **Real-Time Analysis**: Rapid predictions suitable for clinical environments.

---

## System Architecture
The system pipeline includes:
1. **Data Collection** – Liver ultrasound images with corresponding stage labels.
2. **Preprocessing** – Noise reduction, normalization, and edge detection using OpenCV.
3. **Feature Extraction** – ResNet-based CNN to extract detailed image features.
4. **Classification** – Random Forest/XGBoost classifier for fibrosis staging.
5. **Decision Support** – Generates stage predictions with recommended actions.
6. **User Interface** – Web-based portal for uploading images and viewing results.

**Architecture Diagram**  
![Architecture](docs/architecture.png) *(Replace with actual diagram)*

---

## Data Flow
1. Upload liver ultrasound image.
2. Preprocessing enhances image quality.
3. CNN extracts key features.
4. Classifier predicts fibrosis stage (F0–F4).
5. System displays stage, possible causes, and recommended precautions.

---

## Liver Fibrosis Stages (METAVIR)
| Stage | Description | Recommendation |
|-------|-------------|----------------|
| **F0** | No fibrosis | Maintain healthy lifestyle |
| **F1** | Mild fibrosis | Regular monitoring |
| **F2** | Moderate fibrosis | Precautions to prevent progression |
| **F3** | Severe fibrosis | Lifestyle changes, medical consultation |
| **F4** | Cirrhosis | Immediate medical attention |

---

## Technologies Used
- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow / PyTorch
- **Image Processing**: OpenCV
- **Machine Learning**: Random Forest, XGBoost
- **Model Architecture**: ResNet-based CNN
- **Web Framework**: (Flask/Django – specify in implementation)
- **Deployment**: (Local/Cloud – specify in implementation)

---

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/careassist.git
cd careassist

# Install dependencies
pip install -r requirements.txt

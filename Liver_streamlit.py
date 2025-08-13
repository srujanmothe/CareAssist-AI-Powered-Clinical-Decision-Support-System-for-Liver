import streamlit as st
import json
from PIL import Image
import os

# Load stage data
with open("stages.json", "r") as file:
    stages_data = json.load(file)["stages"]

# Streamlit App Configuration
st.set_page_config(page_title="Care Assist - Liver Health Detection", layout="wide")
st.title("🩺 CareAssist: AI-Powered Clinical Decision Support System. ")

# Layout with two columns
col1, col2 = st.columns([1, 2])

with col1:
    # Image Upload
    uploaded_file = st.file_uploader("Upload an image for liver health detection", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        # Save uploaded image to a temporary file
        temp_dir = "uploads"
        os.makedirs(temp_dir, exist_ok=True)
        image_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    if uploaded_file:
        # Simulating a prediction response (Replace this with actual model inference API call)
        detected_stage = "F2"  # Example prediction, replace with actual model output

        st.subheader(f"Detected Stage: {detected_stage}")
        
        if detected_stage in stages_data:
            stage_info = stages_data[detected_stage]
            
            st.markdown("**🛑 Major Causes:**")
            for cause in stage_info["Major Causes"]:
                st.write(f"- {cause}")
            
            st.markdown("**ℹ️ About the Stage:**")
            st.write(stage_info["About the Stage"])
            
            st.markdown("**✅ Preventive Measures:**")
            for measure in stage_info["Preventive Measures"]:
                st.write(f"- {measure}")
            
            st.markdown("**🔄 Reversibility Chances:**")
            for chance in stage_info["Reversibility Chances"]:
                st.write(f"- {chance}")
        else:
            st.error("Stage information not found.")

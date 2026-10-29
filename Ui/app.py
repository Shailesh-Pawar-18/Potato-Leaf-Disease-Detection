import os

# Force TensorFlow to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")

# ---------- Paths (relative to this file, so they work on any PC / server) ----------
# Folder names are case-sensitive on Linux: Ui, Model, Dataset
UI_DIR = os.path.dirname(os.path.abspath(__file__))              # .../Ui
MODEL_PATH = os.path.join(UI_DIR, "..", "Model", "trained_plant_disease_model.keras")
BANNER_PATH = os.path.join(UI_DIR, "diseases.jpg")

# Optional: if the model is NOT stored in GitHub, put its Google Drive file ID here.
# The app will download it automatically the first time it runs.
MODEL_DRIVE_ID = "1AdoXRbWX1_Cqv-wVhBrjCFmFozRa4jB4"

# Same alphabetical order that image_dataset_from_directory used in training
CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

DISEASE_INFO = {
    "Potato___Early_blight": """🛑 **Early Blight**
- **Cause**: Fungus *Alternaria solani*
- **Symptoms**: Dark brown spots with concentric rings on leaves.
- **Prevention**: Avoid overhead watering, use resistant varieties, and apply fungicides.
""",
    "Potato___Late_blight": """⚠️ **Late Blight**
- **Cause**: Pathogen *Phytophthora infestans*
- **Symptoms**: Dark, water-soaked lesions that spread rapidly in humid conditions.
- **Prevention**: Remove infected plants, improve air circulation, and use fungicides.
""",
    "Potato___healthy": """✅ **Healthy Plant**
- Your plant appears **healthy** with no visible disease symptoms. 🎉
- Keep monitoring for any changes and maintain good farming practices!
""",
}


@st.cache_resource
def load_model():
    """Load the model once and reuse it across reruns."""
    if not os.path.exists(MODEL_PATH) and MODEL_DRIVE_ID:
        import gdown
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        gdown.download(id=MODEL_DRIVE_ID, output=MODEL_PATH, quiet=False)
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {os.path.abspath(MODEL_PATH)}")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)


def model_prediction(uploaded_file):
    model = load_model()
    image = Image.open(uploaded_file).convert("RGB").resize((128, 128))
    arr = np.expand_dims(np.array(image, dtype="float32"), axis=0)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return idx, float(preds[idx])


# ---------- Sidebar ----------
st.sidebar.title("🌿 Plant Disease Detection System")
app_mode = st.sidebar.radio("Select Page:", ["Home", "Disease Recognition"])

if os.path.exists(BANNER_PATH):
    st.image(Image.open(BANNER_PATH), width=600)

# ---------- Home ----------
if app_mode == "Home":
    st.markdown(
        "<h1 style='text-align: center;'>🌱 Plant Disease Detection 🌱</h1>",
        unsafe_allow_html=True,
    )

# ---------- Disease Recognition ----------
elif app_mode == "Disease Recognition":
    st.header("🔬 Plant Disease Recognition System")
    test_image = st.file_uploader("📸 Choose an Image:", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📷 Show The Image"):
            if test_image is not None:
                st.image(test_image, use_container_width=True)
            else:
                st.warning("⚠️ Please upload an image first.")

    with col2:
        if st.button("🔍 Predict"):
            if test_image is not None:
                with st.spinner("Analysing image..."):
                    idx, confidence = model_prediction(test_image)
                st.balloons()
                st.write("✅ **Our Prediction**")
                name = CLASS_NAMES[idx]
                st.success(f"🌿 **Model predicts:** {name} ({confidence:.1%} confidence)")
                st.write(DISEASE_INFO[name])
            else:
                st.warning("⚠️ Please upload an image before predicting.")

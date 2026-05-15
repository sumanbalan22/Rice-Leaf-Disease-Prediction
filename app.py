import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

IMG_SIZE = 224
MODEL_PATH = "model/rice_leaf_mobilenetv2.h5"
CLASS_PATH = "model/class_names.npy"

# Force CPU
tf.config.set_visible_devices([], 'GPU')

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = np.load(CLASS_PATH, allow_pickle=True)
    return model, class_names

model, class_names = load_model()

# ------------------- Disease Solutions -------------------

disease_solutions = {
    "bacterial leaf blight": "Use resistant varieties and avoid excess nitrogen fertilizer.\nApply recommended bactericides like copper-based sprays.",
    
    "brown spot": "Apply balanced fertilizers and improve field drainage.\nUse fungicides such as mancozeb if infection is severe.",
    
    "healthy rice leaf": "The leaf is healthy.\nMaintain proper irrigation and balanced fertilization.",
    
    "leaf blast": "Apply fungicides like tricyclazole at early stages.\nAvoid excessive nitrogen fertilizer application.",
    
    "leaf scald": "Use disease-free seeds and practice crop rotation.\nApply appropriate fungicide if symptoms spread.",
    
    "narrow brown leaf spot": "Improve soil fertility and avoid water stress.\nSpray recommended fungicides if required.",
    
    "rice hispa": "Use insecticides like chlorpyrifos if infestation is high.\nRemove and destroy affected leaves.",
    
    "sheath blight": "Maintain proper plant spacing to reduce humidity.\nApply fungicides like validamycin if necessary."
}

# ------------------- Prediction Function -------------------

def predict_image(img):

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    preds = model.predict(img_input)
    idx = np.argmax(preds)
    confidence = preds[0][idx] * 100
    disease = class_names[idx]

    return disease, confidence, img_rgb

# ------------------- Streamlit UI -------------------

st.title("🌾 Rice Leaf Disease Detection (CPU Version)")

uploaded_file = st.file_uploader(
    "Upload Rice Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    disease, confidence, img_rgb = predict_image(img)

    st.image(img_rgb, caption="Uploaded Image")

    disease_lower = disease.lower()

    if "healthy" in disease_lower:
        st.success(f"🌿 Healthy Rice Leaf\nConfidence: {confidence:.2f}%")
    else:
        st.error(f"🦠 Disease: {disease}\nConfidence: {confidence:.2f}%")

    # ----------- Show Solution -----------
    if disease_lower in disease_solutions:
        st.info("💡 Recommended Solution:")
        st.write(disease_solutions[disease_lower])

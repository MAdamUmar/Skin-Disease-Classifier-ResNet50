import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os
import cv2

FILE_ID = "1N6F10miFe4DjOk1scbEhj9z062EPvdTO"
MODEL_FILENAME = "new_resnet50_finetuned.keras"

# Class labels
CLASS_NAMES = [
    "Acne", "Alopecia", "Carcinoma", "Dermatitis", "Dermatofibroma", "Eczema",
    "Healthy", "Keratosis", "Melanoma", "Onychomycosis", "Psoriasis",
    "Rosacea", "Skin_Tag", "Tinea"
]

# Download model if not already present
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILENAME):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_FILENAME, quiet=False)
    model = tf.keras.models.load_model(MODEL_FILENAME)
    return model

model = load_model()

# 🔹 Grad-CAM utility functions
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(original_img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)
    return overlayed_img

st.title("🩺 Skin Disease Classifier")
st.markdown("Upload a skin image, and the model will predict the skin condition.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img)  # keep pixel values 0-255 as ints
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = np.max(predictions)

    st.success(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2%}")


    # 🔹 Grad-CAM Visualization
    # Auto-detect last conv layer (or manually set it)
    try:
        last_conv_layer_name = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
    except IndexError:
        last_conv_layer_name = "conv5_block3_out"  # fallback name for ResNet50

    heatmap = make_gradcam_heatmap(img_array_exp, model, last_conv_layer_name, predicted_index)
    gradcam_img = overlay_gradcam(np.array(image), heatmap)

    st.image(gradcam_img, caption="Grad-CAM Heatmap", use_column_width=True)

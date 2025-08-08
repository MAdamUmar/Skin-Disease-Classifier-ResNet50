import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

from langchain_community.llms import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

# -------------------- HUGGING FACE SETUP --------------------
# Get your Hugging Face API key (replace with your key if not using env var)
HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY")

# Use a small, free model from Hugging Face
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    task="text2text-generation",
    model_kwargs={"temperature": 0.5, "max_length": 200},
    huggingfacehub_api_token=HUGGINGFACE_API_KEY
)

# Prompt template for giving advice
template = """
You are a dermatologist assistant.
The patient has been diagnosed with {disease}.
Give safe, practical, non-prescription advice in plain language.
"""
prompt = PromptTemplate(input_variables=["disease"], template=template)
advice_chain = LLMChain(llm=llm, prompt=prompt)

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

    if predicted_class != "Healthy":
        with st.spinner("Fetching personalized advice..."):
            advice = advice_chain.run(disease=predicted_class)
        st.subheader("📋 Advice")
        st.write(advice)
    else:
        st.info("The skin appears healthy, but if you have symptoms, consult a dermatologist.")

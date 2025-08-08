import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

# NEW IMPORTS
from langchain import PromptTemplate, LLMChain
from langchain_community.llms import HuggingFaceHub

# ----------------- HUGGING FACE SETUP -----------------
# You need to create a free HF account -> get API key -> paste here or set as environment variable
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "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")

# Pick a lightweight, free LLM from Hugging Face
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",  # good, small instruct model
    model_kwargs={"temperature": 0.5, "max_length": 300},
    huggingfacehub_api_token=HUGGINGFACE_API_KEY
)

# Create LangChain prompt for advice
template = """
You are a professional dermatologist assistant.
The model has detected that the patient likely has {disease}.
Provide safe, clear, practical steps the patient can take until they can see a dermatologist.
Avoid giving prescriptions. Write in plain language.
"""


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

     # ----------------- GET ADVICE FROM LLM -----------------
    if predicted_class != "Healthy":
        with st.spinner("Fetching personalized advice..."):
            advice = advice_chain.run(disease=predicted_class)
        st.subheader("📋 Advice")
        st.write(advice)
    else:
        st.info("The skin appears healthy, but if you have symptoms, consult a dermatologist.")

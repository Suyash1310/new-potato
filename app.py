import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('potato_disease_model.h5')

# Define a dictionary to label all the plant disease classes
classes = ['Healthy', 'Late Blight', 'Early Blight']

def model_predict(img, model):
    img = img.resize((150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    preds = model.predict(img)
    return preds

# Streamlit interface
st.title("Potato Disease Classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    preds = model_predict(img, model)
    pred_class = classes[np.argmax(preds)]
    st.write(f"Prediction: {pred_class}")

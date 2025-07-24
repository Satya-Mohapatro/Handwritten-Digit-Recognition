
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import joblib
best_knn_pipeline = joblib.load('knn_model.pkl')

st.title(" MNIST Handwritten Digit Recognition using KNN")
st.write("Upload a 28x28 digit image or draw on paper and upload a clear photo for digit prediction.")

uploaded_file = st.file_uploader("Upload a digit image (PNG/JPG):", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('L')  
    st.image(img, caption="Uploaded Image", width=150)

    img = ImageOps.invert(img)
    img_resized = img.resize((28, 28))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_flat = img_array.reshape(1, -1)

    prediction = best_knn_pipeline.predict(img_flat)[0]
    st.subheader(f"✅ Predicted Digit: **{prediction}**")

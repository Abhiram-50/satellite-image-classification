import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd

# Load trained model
model = tf.keras.models.load_model("model.h5")

# Class labels
classes = [
    "Desert",
    "Forest",
    "Highway",
    "Industrial",
    "Residential",
    "River",
    "SeaLake"
]

# Page Settings
st.set_page_config(
    page_title="Satellite Image Classification",
    page_icon="🛰️",
    layout="wide"
)

# Title
st.title("🛰️ Deep Learning Based Satellite Image Classification")

st.markdown("""
Upload a satellite image and the CNN model will classify it into one of the land cover categories.
""")

# Sidebar
st.sidebar.header("Project Information")

st.sidebar.write("""
**Algorithm:** CNN

**Classes:**
- Desert
- Forest
- Highway
- Industrial
- Residential
- River
- SeaLake

**Image Size:** 128 × 128
""")

# Upload Image
uploaded_file = st.file_uploader(
    "Upload Satellite Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    resized_image = image.resize((128,128))

    with col2:
        st.subheader("Resized Image")
        st.image(resized_image, use_container_width=True)

    # Convert to array
    img_array = np.array(resized_image)

    # Normalize
    normalized_image = img_array / 255.0

    st.subheader("Normalized Image")
    st.image(normalized_image)

    # Prepare Input
    input_image = np.expand_dims(
        normalized_image,
        axis=0
    )

    # Prediction
    prediction = model.predict(
        input_image,
        verbose=0
    )

    predicted_class = classes[
        np.argmax(prediction)
    ]

    confidence = np.max(prediction) * 100

    # Result
    st.subheader("Prediction Result")

    st.success(
        f"Predicted Class: {predicted_class}"
    )

    st.info(
        f"Confidence Score: {confidence:.2f}%"
    )

    # Probability Chart
    st.subheader("Class Probabilities")

    probabilities = prediction[0] * 100

    df = pd.DataFrame({
        "Class": classes,
        "Probability (%)": probabilities
    })

    st.bar_chart(
        df.set_index("Class")
    )

    # Probability Table
    st.dataframe(df)
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import streamlit as st
import io

# Constants
IMAGE_SIZE = 192

cnn_model = tf.keras.models.load_model("static/models/dog_cat_M.keras")


# Set up the Streamlit app
st.title("Dog & Cat Classifier")
st.write("Upload an image of a pet to classify it as either a dog or a cat.")

# Preprocess an image
def preprocess_image(image):
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0  # normalize to [0,1] range
    return image

# Classify image function
def classify(model, image):
    image = preprocess_image(image)
    image = tf.reshape(image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
    prob = model.predict(image)
    label = "Cat" if prob[0][0] >= 0.5 else "Dog"
    classified_prob = prob[0][0] if prob[0][0] >= 0.5 else 1 - prob[0][0]
    return label, classified_prob

# Image upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded file once
    file_bytes = uploaded_file.read()

    # Display uploaded image
    image = Image.open(io.BytesIO(file_bytes))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to tensor and classify
    image = tf.image.decode_image(file_bytes, channels=3)
    label, prob = classify(cnn_model, image)
    prob = round((prob * 100), 2)

    # Display result
    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: **{prob}%**")

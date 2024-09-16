import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# Load the models
lungs_model = tf.keras.models.load_model('D:/FinalProjectML/savedmodels/lung_model.keras')
colon_model = tf.keras.models.load_model('D:/FinalProjectML/savedmodels/colon_model.keras')



# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System', 
                           ['Lungs Prediction', 'Colon Prediction'],
                           icons=['lungs', 'capsule'],
                           default_index=0)

# Lungs Model Page
if selected == 'Lungs Prediction':
    st.title("Lungs Disease Prediction System")
    lungs_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if lungs_image is not None:
        st.image(lungs_image, caption='Uploaded Image.', use_column_width=True)

        if st.button("Classify Lungs Image"):
            try:
                image = Image.open(lungs_image).convert('RGB')  # Ensure image is in RGB format
                img = np.array(image)
                img_resized = cv2.resize(img, (80, 80))  # Resize to match model input
                img_resized_scaled = img_resized / 255.0  # Normalize pixel values
                img_array = np.expand_dims(img_resized_scaled, axis=0)  # Add batch dimension

                predictions = lungs_model.predict(img_array)
                score = tf.nn.softmax(predictions[0])
                predicted_class = np.argmax(score)
                class_names = ['lung_n', 'lung_aca', 'lung_scc']  # Make sure these match your model's output

                st.write(f"Prediction: {class_names[predicted_class]}")
                
            except Exception as e:
                st.write(f"Error during prediction: {e}")

# Colon Model Page
if selected == 'Colon Prediction':
    st.title("Colon Disease Prediction System")

    colon_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if colon_image is not None:
        st.image(colon_image, caption='Uploaded Image.', use_column_width=True)

        if st.button("Classify Colon Image"):
            try:
                image = Image.open(colon_image).convert('RGB')  # Ensure image is in RGB format
                img = np.array(image)
                img_resized = cv2.resize(img, (80, 80))  # Resize to match model input
                img_resized_scaled = img_resized / 255.0  # Normalize pixel values
                img_array = np.expand_dims(img_resized_scaled, axis=0)  # Add batch dimension

                predictions = colon_model.predict(img_array)
                score = tf.nn.softmax(predictions[0])
                predicted_class = np.argmax(score)
                class_names = ['colon_n', 'colon_aca']  # Make sure these match your model's output

                st.write(f"Prediction: {class_names[predicted_class]}")
                
            except Exception as e:
                st.write(f"Error during prediction: {e}")





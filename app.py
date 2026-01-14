import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import load_img
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

st.set_page_config(page_title="Brain Tumor Detector", layout="centered")

st.title("Welcome to my School Science Project")
st.write("By: Arnav")
st.write("Class: 10")

st.write("---")
st.header("Project Information")
st.write("This app uses a Convolutional Neural Network (CNN) to look at MRI scans.")
st.write("The model was trained on thousands of images to learn what a tumor looks like.")
st.write("Goal: To help doctors identify brain tumors faster.")

st.write("---")
st.header("Step 1: Upload your MRI Scan")
st.write("Please upload a JPG or PNG file below.")

uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.info("File uploaded successfully!")
    st.image(uploaded_file, caption="The Image you uploaded", use_container_width=True)
    
    st.write("Running preprocessing...")
    my_image = load_img(uploaded_file)
    my_image = my_image.resize((224, 224))
    image_array = img_to_array(my_image)
    image_array = image_array / 255.0
    final_input = np.expand_dims(image_array, axis=0)
    st.write("Image resized to 224x224 and normalized.")

    if st.button("Run Model Prediction"):
        st.write("Loading model...")
        my_model = tf.keras.models.load_model("brain_tumor_binary.h5")
        st.write("Model loaded. Finding result...")
        
        prediction_value = my_model.predict(final_input)[0][0]
        
        st.write("---")
        if prediction_value >= 0.5:
            st.error("RESULT: TUMOR DETECTED")
            st.write("The model is very sure this is a tumor.")
            st.write("Probability Score:")
            st.write(prediction_value)
        else:
            st.success("RESULT: NO TUMOR DETECTED")
            st.write("The scan looks clear of tumors.")
            st.write("Probability Score:")
            st.write(1 - prediction_value)
        st.write("---")

st.header("Step 2: Model Performance and Graphs")
st.write("This section shows how the model performed on the test data folder.")

if st.button("Calculate Metrics"):
    st.write("Accessing the Testing folder on GitHub...")
    
    if os.path.exists("Testing"):
        st.write("Folder found! Starting bulk prediction...")
        
        test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            "Testing",
            image_size=(224, 224),
            batch_size=1,
            shuffle=False
        )
        
        actual_list = []
        predicted_list = []
        
        count = 0
        for images, labels in test_dataset:
            count = count + 1
            raw_pred = my_model.predict(images, verbose=0)[0][0]
            
            real_val = labels.numpy()[0]
            actual_list.append(real_val)
            
            if raw_pred >= 0.5:
                predicted_list.append(1)
            else:
                predicted_list.append(0)
        
        st.

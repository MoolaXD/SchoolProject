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
import time

st.set_page_config(page_title="Brain Tumor Detector", layout="centered")

st.title("My School Personal Project: Brain Tumor Detector")
st.write("Created by: Arnav")
st.write("Date: 2025")

st.sidebar.title("Menu")
selection = st.sidebar.selectbox("Go to page:", ["Home", "Detector", "Graphs and Math"])

if selection == "Home":
    st.header("Project Overview")
    st.write("I made this project to help detect brain tumors from MRI scans.")
    st.write("I am using a CNN model that I trained myself.")
    st.write("The model checks if an image has a tumor or not.")
    st.write("---")
    st.write("How to use:")
    st.write("1. Click the Menu on the left.")
    st.write("2. Go to 'Detector' to upload an image.")
    st.write("3. Go to 'Graphs' to see the accuracy.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/5f/MRI_brain.jpg", caption="Example of an MRI")

if selection == "Detector":
    st.header("Brain Tumor Detector")
    st.write("Upload a JPG file of a brain scan below:")
    
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    if file is not None:
        st.write("File detected!")
        st.image(file, width=300)
        
        st.write("Loading the model file brain_tumor_binary.h5.")
        try:
            my_model = tf.keras.models.load_model("brain_tumor_binary.h5")
            st.write("Model loaded successfully.")
        except:
            st.write("Error: Could not load the model file.")

        st.write("Preparing the image for the model.")
        temp_img = load_img(file)
        temp_img = temp_img.resize((224, 224))
        arr = img_to_array(temp_img)
        arr = arr / 255.0
        ready_img = np.expand_dims(arr, axis=0)
        st.write("Resizing done. Normalizing done.")

        if st.button("Predict Now"):
            st.write("Thinking.")
            time.sleep(1)
            output = my_model.predict(ready_img)[0][0]
            
            if output >= 0.5:
                st.subheader("Result: TUMOR FOUND")
                st.write("Confidence level:")
                st.write(output)
            else:
                st.subheader("Result: NO TUMOR")
                st.write("Confidence level:")
                st.write(1 - output)

if selection == "Graphs and Math":
    st.header("Performance Results")
    st.write("Testing the model on my lite testing dataset folder.")
    
    if st.button("Run Accuracy Test"):
        st.write("Checking if Testing folder exists.")
        
        if os.path.exists("Testing"):
            st.write("Folder found.")
            
            try:
                m = tf.keras.models.load_model("brain_tumor_binary.h5")
                
                data = tf.keras.preprocessing.image_dataset_from_directory(
                    "Testing",
                    image_size=(224, 224),
                    batch_size=1,
                    shuffle=False
                )
                
                names = data.class_names
                st.write("Classes found: " + str(names))
                
                true_vals = []
                pred_vals = []
                
                st.write("Predicting images one by one.")
                bar = st.progress(0)
                
                all_images = list(data)
                total = len(all_images)
                
                for i in range(total):
                    img_batch, label_batch = all_images[i]
                    p = m.predict(img_batch, verbose=0)[0][0]
                    
                    real = label_batch.numpy()[0]
                    true_vals.append(real)
                    
                    if p >= 0.5:
                        pred_vals.append(1)
                    else:
                        pred_vals.append(0)
                    
                    bar.progress((i + 1) / total)

                st.write("Calculations finished.")
                
                st.write("---")
                st.write("CONFUSION MATRIX")
                cm = confusion_matrix(true_vals, pred_vals)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                st.pyplot(fig)
                
                st.write("---")
                st.write("FULL STATISTICS TABLE")
                rep = classification_report(true_vals, pred_vals, output_dict=True)
                st.table(pd.DataFrame(rep).transpose())
                
                st.write("Final Conclusion: The project worked!")
                
            except Exception as e:
                st.write("Something went wrong during the math:")
                st.write(e)
        else:
            st.error("FileNotFoundError: The 'Testing' folder is not in GitHub!")
            st.write("I need to make sure the folder is named 'Testing' and has images inside.")


import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import load_img
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

MODEL_PATH = r"brain_tumor_binary.h5"
TEST_PATH = r"Testing"

model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["No Tumor", "Tumor"]

def preprocess(img):
    img = img.resize((224, 224))
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

def load_test_data():
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_PATH,
        image_size=(224, 224),
        batch_size=1,
        shuffle=False
    )
    return ds, ds.class_names

@st.cache_data
def compute_metrics():
    test_ds, class_names = load_test_data()

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)[0][0]

        true_class = class_names[labels.numpy()[0]]
        y_true.append(0 if true_class == "notumor" else 1)

        y_pred.append(1 if preds >= 0.5 else 0)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True
    )

    return cm, report

st.title("Brain Tumor Detection System")

tab1, tab2 = st.tabs(["Prediction", "Model Performance"])

with tab1:
    uploaded_file = st.file_uploader(
        "Upload Brain MRI Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)
        img = load_img(uploaded_file)

        if st.button("Predict"):
            processed = preprocess(img)
            prob = model.predict(processed)[0][0]

            if prob >= 0.5:
                st.subheader("Prediction: Tumor Detected")
                st.write(f"Confidence: {prob:.4f}")
            else:
                st.subheader("Prediction: No Tumor")
                st.write(f"Confidence: {1 - prob:.4f}")

with tab2:
    cm, report = compute_metrics()

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax
    )
    st.pyplot(fig)

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [
            report["accuracy"],
            report["Tumor"]["precision"],
            report["Tumor"]["recall"],
            report["Tumor"]["f1-score"]
        ]
    })

    st.table(metrics_df)


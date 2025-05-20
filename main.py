import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
import tensorflow.keras.backend as k


@register_keras_serializable()
def recall_class_2(y_true, y_pred):
    class_id = 2
    y_true_class = k.argmax(y_true, axis=-1)
    y_pred_class = k.argmax(y_pred, axis=-1)

    true_positives = k.sum(k.cast((y_true_class == class_id) & (y_pred_class == class_id), dtype='float32'))
    possible_positives = k.sum(k.cast(y_true_class == class_id, dtype='float32'))

    recall = true_positives / (possible_positives + k.epsilon())
    return recall


model_file = 'model8.keras'
model = load_model(model_file, custom_objects={'recall_class_2': recall_class_2})

label_map = {0: 'Cat', 1: 'Dog', 2: 'Snake'}

st.title("Image Classification - Cat vs Dog vs Snake")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image = image.resize((64, 64))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx]

    st.write(f"### Prediction: {label_map[class_idx]} ({confidence * 100:.2f}% confidence)")
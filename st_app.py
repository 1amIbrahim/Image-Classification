import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from skimage.transform import resize
from Bsai23046_Mini_Project import forward_pass, sigmoid
from skimage.filters import gaussian

weights = []
for i in range(4):
    w = np.load(f"weight_layer_{i}.npy")
    weights.append(w)

def predict_digit(img, weights):
    activations, _ = forward_pass(img, weights, sigmoid)
    preds = activations[-1]  # Softmax outputs
    top2 = np.argsort(preds[0])[-2:][::-1]  # Get top 2 predictions
    return top2, preds[0][top2[0]], preds[0][top2[1]]

from scipy.ndimage import center_of_mass, shift

def center_image(img):
    cy, cx = center_of_mass(img)
    shift_y = 14 - cy
    shift_x = 14 - cx
    return shift(img, shift=(shift_y, shift_x), mode='constant', cval=0)


st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")
st.title("✏️ MNIST Digit Classifier - Draw Your Digit")

col1, col2 = st.columns([2, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    if st.button("Clear Canvas"):
        st.rerun()

if canvas_result.image_data is not None:
    img = canvas_result.image_data
    img = np.array(img)
    img = img[:, :, 0]  # Grayscale
    img = 255 - img  # Invert
    img = img / 255.0  # Normalize

    img = gaussian(img, sigma=1)  # Optional: make strokes softer
    img_resized = resize(img, (28, 28), anti_aliasing=True)
    img_resized[img_resized < 0.3] = 0
    img_resized[img_resized >= 0.3] = 1

    img_resized = center_image(img_resized)

    st.subheader("28x28 Image Given to Model")
    st.image(img_resized, width=150, clamp=True)

    img_flatten = img_resized.flatten().reshape(1, -1)
    top2, prob1, prob2 = predict_digit(img_flatten, weights)

    if np.sum(img_resized) > 10:  # Only predict if something was drawn
        st.success(f"Prediction 1️⃣: {top2[0]} (Confidence: {prob1:.2f})")
        st.info(f"Maybe 2️⃣: {top2[1]} (Confidence: {prob2:.2f})")
    else:
        st.warning("Draw something to get a prediction!")

import streamlit as st
import numpy as np
import cv2
import joblib
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# --- Page Config ---
st.set_page_config(page_title="Zenith Cluster", layout="centered")

st.title("🚀 EE5130: Zenith Cluster Exhibition")
st.caption("Developed by: Sakith Nethkalana")
st.write("Draw digits on the canvas below and click **Predict**.")

# --- Load Models (Cached for speed) ---
@st.cache_resource
def load_models():
    rf = joblib.load('rf_model.pkl')
    mlp = joblib.load('mlp_model.pkl')
    cnn = tf.keras.models.load_model('cnn_model.h5')
    pca = joblib.load('pca_100.pkl')
    return rf, mlp, cnn, pca

with st.spinner("Loading AI Models..."):
    rf_model, mlp_model, cnn_model, pca_100 = load_models()

# --- Preprocessing Functions ---
def preprocess_digit_mnist_standard(digit_crop):
    h, w = digit_crop.shape
    size = max(h, w) + 8
    square_digit = np.zeros((size, size), dtype=np.uint8)
    pad_y, pad_x = (size - h) // 2, (size - w) // 2
    square_digit[pad_y:pad_y+h, pad_x:pad_x+w] = digit_crop
    square_digit = cv2.GaussianBlur(square_digit, (3, 3), 0)
    digit_20 = cv2.resize(square_digit, (20, 20), interpolation=cv2.INTER_AREA)
    padded = np.zeros((28, 28), dtype=np.float32)
    padded[4:24, 4:24] = digit_20 / 255.0
    M = cv2.moments(padded)
    if M["m00"] > 0:
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        shift_x, shift_y = np.clip(np.round(14 - cx), -4, 4), np.clip(np.round(14 - cy), -4, 4)
        M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        padded = cv2.warpAffine(padded, M_shift, (28, 28))
    return padded

def extract_digit_bounding_boxes(img_np):
    contours, _ = cv2.findContours(img_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if h < 12 or w < 5 or area < 40: continue
        aspect_ratio = w / float(h)
        if aspect_ratio > 2.0:
            split_w = w // 3
            bounding_boxes.extend([(x, y, split_w, h), (x + split_w, y, split_w, h), (x + 2 * split_w, y, split_w, h)])
        elif aspect_ratio > 1.25:
            split_w = w // 2
            bounding_boxes.extend([(x, y, split_w, h), (x + split_w, y, split_w, h)])
        else:
            bounding_boxes.append((x, y, w, h))
    return sorted(bounding_boxes, key=lambda b: b[0])

# --- UI Canvas ---
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    height=200,
    width=500,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict Canvas", type="primary"):
    if canvas_result.image_data is not None:
        img_rgba = canvas_result.image_data
        img_gray = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2GRAY)
        
        if np.max(img_gray) == 0:
            st.warning("Please draw something on the canvas first!")
        else:
            boxes = extract_digit_bounding_boxes(img_gray)
            if not boxes:
                st.warning("No valid digits detected!")
            else:
                rf_digits, mlp_digits, cnn_digits, ens_digits = [], [], [], []
                for (x, y, w, h) in boxes:
                    digit_crop = img_gray[y:y+h, x:x+w]
                    digit_norm = preprocess_digit_mnist_standard(digit_crop)
                    
                    flat_pca = pca_100.transform(digit_norm.reshape(1, 784))
                    cnn_input = digit_norm.reshape(1, 28, 28, 1)

                    rf_prob = rf_model.predict_proba(flat_pca)[0]
                    mlp_prob = mlp_model.predict_proba(flat_pca)[0]
                    cnn_prob = cnn_model.predict(cnn_input, verbose=0)[0]

                    rf_digits.append(str(np.argmax(rf_prob)))
                    mlp_digits.append(str(np.argmax(mlp_prob)))
                    cnn_digits.append(str(np.argmax(cnn_prob)))

                    ens_prob = (rf_prob * 0.25) + (mlp_prob * 0.25) + (cnn_prob * 0.50)
                    ens_digits.append(str(np.argmax(ens_prob)))

                st.success(f"**🏆 Ensemble Voting Output (98.94% Acc):** {''.join(ens_digits)}")
                col1, col2, col3 = st.columns(3)
                col1.info(f"🌲 Random Forest: {''.join(rf_digits)}")
                col2.info(f"🧠 MLP Neural Net: {''.join(mlp_digits)}")
                col3.info(f"📷 TensorFlow CNN: {''.join(cnn_digits)}")

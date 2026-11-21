import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from features import N_FEATURES, parse_pe, raw_features, summary, vectorize

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "Virus_detect.keras"
SCALER_PATH = APP_DIR / "scaler.json"

# Output column of the model that means "malicious" (0 = benign, 1 = malicious).
MALICIOUS_CLASS = 1


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_resource
def load_scaler():
    with open(SCALER_PATH) as f:
        data = json.load(f)
    if data["n_features"] != N_FEATURES:
        raise ValueError("scaler.json does not match the features in features.py")
    return np.array(data["mean"]), np.array(data["scale"])


def prepare_input(raw, mean, scale):
    """Vectorize and standardize features the same way as in training."""
    x = vectorize(raw).astype(np.float64)
    return ((x - mean) / scale)[np.newaxis, :].astype(np.float32)


# Streamlit UI
st.set_page_config(
    page_title="Emguard Virus Detection",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stButton>button { background-color: #007bff; color: white; border-radius: 5px; }
    .result { font-size: 18px; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛡️ Emguard Virus Detection")
st.write("Upload your PE file to check if it's safe or malicious.")

try:
    model = load_model()
    preprocessing = load_scaler()
except Exception as e:
    st.error(f"Could not load the detection model: {e}", icon="⚠️")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a Portable Executable (PE) file (.exe, .dll)",
    type=["exe", "dll"],
    help="Only files with .exe and .dll extensions are supported.",
)

if uploaded_file:
    bytez = uploaded_file.getvalue()

    if not bytez:
        st.error("The uploaded file is empty.", icon="⚠️")
        st.stop()

    with st.spinner("Analyzing the file..."):
        binary = parse_pe(bytez)
        if binary is None:
            st.error(
                "This file is not a valid Portable Executable (PE) file, so it cannot be analyzed.",
                icon="⚠️",
            )
            st.stop()
        raw = raw_features(bytez, binary)
        probs = model.predict(prepare_input(raw, *preprocessing), verbose=0)[0]

    malicious_prob = float(probs[MALICIOUS_CLASS])

    st.subheader("🔍 Detection Results")
    if malicious_prob > 0.5:
        st.error(
            f"The uploaded file is **Malicious** with a confidence of **{malicious_prob * 100:.2f}%**.",
            icon="🚨",
        )
    else:
        st.success(
            f"The uploaded file is **Safe** with a confidence of **{(1 - malicious_prob) * 100:.2f}%**.",
            icon="✅",
        )

    with st.expander("Extracted features"):
        st.json(summary(raw))

st.caption(
    "Emguard is a machine-learning model and can make mistakes. "
    "Do not rely on it as your only antivirus check."
)

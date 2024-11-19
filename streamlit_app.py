import streamlit as st
import numpy as np
import tensorflow as tf
import lief

# Load pre-trained model
MODEL_PATH = "Virus_detect.h5"  # Adjust to the correct path of your uploaded model
model = tf.keras.models.load_model(MODEL_PATH)

# Feature extraction classes (same as before)
class ByteHistogram:
    def feature_vector(self, bytez):
        counts = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256)
        return counts / counts.sum()

class ByteEntropyHistogram:
    def __init__(self, step=1024, window=2048):
        self.window = window
        self.step = step

    def _entropy_bin_counts(self, block):
        c = np.bincount(block >> 4, minlength=16)
        p = c.astype(np.float32) / self.window
        wh = np.where(c)[0]
        H = np.sum(-p[wh] * np.log2(p[wh])) * 2
        return (int(H * 2) if H < 8.0 else 15), c

    def feature_vector(self, bytez):
        output = np.zeros((16, 16), dtype=int)
        a = np.frombuffer(bytez, dtype=np.uint8)
        if a.shape[0] < self.window:
            Hbin, c = self._entropy_bin_counts(a)
            output[Hbin, :] += c
        else:
            shape = a.shape[:-1] + (a.shape[-1] - self.window + 1, self.window)
            strides = a.strides + (a.strides[-1],)
            blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::self.step, :]
            for block in blocks:
                Hbin, c = self._entropy_bin_counts(block)
                output[Hbin, :] += c
        return output.flatten() / output.sum()

class SectionInfo:
    def feature_vector(self, binary):
        section_sizes = [section.size for section in binary.sections]
        section_entropy = [section.entropy for section in binary.sections]
        section_sizes = np.pad(section_sizes, (0, 10 - len(section_sizes)), 'constant')[:10]
        section_entropy = np.pad(section_entropy, (0, 10 - len(section_entropy)), 'constant')[:10]
        return np.concatenate([section_sizes, section_entropy])

class PEFeatureExtractor:
    def extract_features(self, bytez):
        byte_hist = ByteHistogram().feature_vector(bytez)
        byte_entropy_hist = ByteEntropyHistogram().feature_vector(bytez)
        lief_binary = lief.parse(raw=bytez)

        if lief_binary:
            section_features = SectionInfo().feature_vector(lief_binary)
            imports_count = len(lief_binary.imports)
            exports_count = len(lief_binary.exported_functions)
            has_debug = int(lief_binary.has_debug)
        else:
            section_features = np.zeros(20)
            imports_count = 0
            exports_count = 0
            has_debug = 0

        histogram_mean = byte_hist.mean()
        histogram_std = byte_hist.std()
        byteentropy_mean = byte_entropy_hist.mean()
        byteentropy_std = byte_entropy_hist.std()

        features = np.concatenate([
            section_features,
            [histogram_mean, histogram_std, byteentropy_mean, byteentropy_std],
            [imports_count, exports_count, has_debug]
        ])
        return features

# Streamlit UI
st.set_page_config(
    page_title="Emguard Virus Detection",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Header
st.markdown(
    """
    <style>
    .main { background-color: #f9f9f9; }
    .stButton>button { background-color: #007bff; color: white; border-radius: 5px; }
    .stFileUploader>div { background-color: #e9ecef; border-radius: 5px; }
    .result { font-size: 18px; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🛡️ Emguard Virus Detection")
st.write("Upload your PE file to check if it's safe or malicious.")

# File uploader
uploaded_file = st.file_uploader(
    "Upload a Portable Executable (PE) file (.exe, .dll)",
    type=["exe", "dll"],
    help="Only files with .exe and .dll extensions are supported."
)

if uploaded_file:
    bytez = uploaded_file.read()
    with st.spinner("Analyzing the file..."):
        extractor = PEFeatureExtractor()
        feature_vector = extractor.extract_features(bytez)

    if feature_vector is not None:
        expected_columns = model.input_shape[-1]

        if len(feature_vector) > expected_columns:
            feature_vector = feature_vector[:expected_columns]
        elif len(feature_vector) < expected_columns:
            feature_vector = np.pad(feature_vector, (0, expected_columns - len(feature_vector)), 'constant')

        prediction = model.predict(np.expand_dims(feature_vector, axis=0))[0]
        prediction_value = prediction[0] if isinstance(prediction, (np.ndarray, list)) else prediction

        st.subheader("🔍 Detection Results")
        if prediction_value > 0.5:
            st.error(
                f"The uploaded file is **Malicious** with a confidence of **{prediction_value * 100:.2f}%**.",
                icon="🚨"
            )
        else:
            st.success(
                f"The uploaded file is **Safe** with a confidence of **{(1 - prediction_value) * 100:.2f}%**.",
                icon="✅"
            )
    else:
        st.error("Unable to extract features from the uploaded file. Please ensure the file is valid.", icon="⚠️")

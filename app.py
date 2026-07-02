import streamlit as st
from PIL import Image
import tempfile

from src.model import ScreenDetector

st.set_page_config(
    page_title="Spot the Fake Photo",
    page_icon="📸",
    layout="wide"
)

@st.cache_resource
def load_model():
    return ScreenDetector()

detector = load_model()

st.title("📸 Spot the Fake Photo")

st.write(
    "Detect whether an image is a **real photograph** "
    "or a **photo of a screen**."
)

tab1, tab2 = st.tabs(["📁 Upload", "📷 Camera"])

uploaded = tab1.file_uploader(
    "Choose Image",
    type=["jpg", "jpeg", "png"]
)

camera = tab2.camera_input("Take Photo")

image_file = uploaded if uploaded else camera

if image_file:

    image = Image.open(image_file)

    col1, col2 = st.columns([2,1])

    with col1:

        st.image(image, use_container_width=True)

    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as tmp:

        image.save(tmp.name)

        score, latency = detector.predict_timed(tmp.name)

    confidence = score if score > 0.5 else 1-score

    with col2:

        st.subheader("Prediction")

        if score >= 0.5:

            st.error("📺 SCREEN RECAPTURE")

        else:

            st.success("📷 REAL PHOTO")

        st.metric(
            "Confidence",
            f"{confidence*100:.1f}%"
        )

        st.progress(float(confidence))

        st.metric(
            "Latency",
            f"{latency:.1f} ms"
        )

        st.metric(
            "Screen Probability",
            f"{score:.3f}"
        )

        st.metric(
            "Real Probability",
            f"{1-score:.3f}"
        )
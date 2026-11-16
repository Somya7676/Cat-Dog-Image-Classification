
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
import numpy as np

from PIL import Image


# ============================================
# PAWSENSE AI - CONFIGURATION
# ============================================

st.set_page_config(
    page_title="PawSense AI | Cat & Dog Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "cat_dog_model.keras"
CSS_PATH = BASE_DIR / "static" / "style.css"
JS_PATH = BASE_DIR / "static" / "script.js"

IMG_SIZE = (160, 160)


# ============================================
# LOAD CSS
# ============================================

if CSS_PATH.exists():
    css = CSS_PATH.read_text(encoding="utf-8")

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True
    )
else:
    st.warning("CSS file not found in static folder.")


# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(
        "Trained model not found! "
        "Please check models/cat_dog_model.keras"
    )
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load trained model: {error}")
    st.stop()


# ============================================
# SESSION STATE
# ============================================

if "history" not in st.session_state:
    st.session_state.history = []

if "latest" not in st.session_state:
    st.session_state.latest = None


# ============================================
# SIDEBAR
# ============================================

with st.sidebar:

    st.markdown(
        '<div class="brand">🐾 PawSense <span>AI</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-caption">INTELLIGENT PET VISION</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### WORKSPACE")

    page = st.radio(
        "Navigation",
        ["🏠 Image Classifier", "📘 About Project"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### MODEL INFORMATION")

    st.caption("Architecture")
    st.write("MobileNetV2")

    st.caption("Learning Technique")
    st.write("Transfer Learning")

    st.caption("Image Classes")
    st.write("🐱 Cat  ·  🐶 Dog")

    st.caption("Input Resolution")
    st.write("160 × 160 pixels")

    st.divider()

    st.success("AI MODEL READY")


# ============================================
# TOP HEADER
# ============================================

left_header, right_header = st.columns([3, 1])

with left_header:
    st.caption("DEEP LEARNING • IMAGE CLASSIFICATION")

with right_header:
    st.success("● SYSTEM ONLINE")


# ============================================
# IMAGE CLASSIFIER PAGE
# ============================================

if page == "🏠 Image Classifier":

    st.markdown(
        '<div class="hero-tag">AI POWERED IMAGE RECOGNITION</div>',
        unsafe_allow_html=True
    )

    st.title("Who's in the picture?")

    st.markdown(
        '<div class="hero-description">'
        'Upload a photo and let PawSense AI identify your furry friend. '
        'Discover intelligent image classification powered by '
        'deep learning and transfer learning.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")
    st.divider()

    upload_col, result_col = st.columns(
        [1.1, 0.9],
        gap="large"
    )

    # ----------------------------------------
    # IMAGE UPLOAD
    # ----------------------------------------

    with upload_col:

        st.subheader("📤 Upload your image")

        st.caption(
            "Choose a clear photo of a cat or dog."
        )

        uploaded_file = st.file_uploader(
            "Select image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG and PNG"
        )

        if uploaded_file is not None:

            try:
                image = Image.open(uploaded_file).convert("RGB")

                st.image(
                    image,
                    caption="Uploaded image preview",
                    use_container_width=True
                )

                if st.button(
                    "✦ Analyse Image",
                    type="primary",
                    use_container_width=True
                ):

                    with st.spinner(
                        "PawSense AI is analysing your image..."
                    ):

                        # Prepare image
                        resized = image.resize(
                            IMG_SIZE,
                            Image.Resampling.BILINEAR
                        )

                        arr = np.asarray(
                            resized,
                            dtype=np.float32
                        )

                        arr = np.expand_dims(arr, axis=0)

                        # Model prediction
                        prediction = model.predict(
                            arr,
                            verbose=0
                        )

                        dog_probability = float(
                            np.asarray(prediction).reshape(-1)[0]
                        )

                        dog_probability = float(
                            np.clip(dog_probability, 0.0, 1.0)
                        )

                        cat_probability = 1.0 - dog_probability

                        if dog_probability >= 0.5:
                            label = "Dog"
                            confidence = dog_probability
                            emoji = "🐶"
                        else:
                            label = "Cat"
                            confidence = cat_probability
                            emoji = "🐱"

                        result = {
                            "label": label,
                            "confidence": confidence,
                            "emoji": emoji,
                            "filename": uploaded_file.name
                        }

                        st.session_state.latest = result

                        st.session_state.history.insert(
                            0,
                            result.copy()
                        )

                        st.session_state.history = (
                            st.session_state.history[:5]
                        )

                    st.success("Analysis completed!")

            except Exception as error:
                st.error(f"Image processing error: {error}")

        else:
            st.info(
                "👆 Upload an image to begin your AI prediction."
            )

    # ----------------------------------------
    # PREDICTION RESULT
    # ----------------------------------------

    with result_col:

        st.subheader("✨ AI Prediction")

        result = st.session_state.latest

        if result is not None:

            st.markdown(
                f"## {result['emoji']} {result['label']}"
            )

            st.metric(
                "CONFIDENCE SCORE",
                f"{result['confidence'] * 100:.2f}%"
            )

            st.progress(
                result["confidence"],
                text="Model confidence"
            )

            st.caption(
                f"Image: {result['filename']}"
            )

            st.markdown("#### Prediction summary")

            st.write(
                f"The trained model classified the uploaded "
                f"image as **{result['label']}**."
            )

            # JavaScript copy component
            js_code = ""

            if JS_PATH.exists():
                js_code = JS_PATH.read_text(encoding="utf-8")

            safe_label = result["label"]
            safe_confidence = f"{result['confidence'] * 100:.2f}%"

            copy_html = f"""
            <!DOCTYPE html>
            <html>
            <body style="
                background: transparent;
                font-family: Arial, sans-serif;
                color: white;
            ">

            <button id="copyButton" style="
                background: linear-gradient(100deg,#7c3aed,#4f46e5);
                color: white;
                border: 0;
                border-radius: 10px;
                padding: 13px 20px;
                cursor: pointer;
                font-weight: bold;
                font-size: 14px;
            ">
                ⧉ Copy Prediction
            </button>

            <span id="copyStatus" style="
                font-size: 12px;
                margin-left: 10px;
            "></span>

            <script>
            {js_code}

            document.getElementById("copyButton")
                .addEventListener("click", function() {{

                    window.copyPawSenseResult(
                        "{safe_label}",
                        "{safe_confidence}"
                    );

                }});
            </script>

            </body>
            </html>
            """

            components.html(
                copy_html,
                height=65,
                scrolling=False
            )

        else:

            st.info(
                "Your prediction will appear here after image analysis."
            )

            st.markdown("### What happens next?")

            st.write("🐾 Upload a cat or dog image.")
            st.write("🧠 AI analyses the image.")
            st.write("✨ View the prediction and confidence score.")

    # ----------------------------------------
    # RECENT PREDICTIONS
    # ----------------------------------------

    st.divider()

    st.subheader("🕘 Recent Predictions")

    if st.session_state.history:

        history_columns = st.columns(
            len(st.session_state.history)
        )

        for column, item in zip(
            history_columns,
            st.session_state.history
        ):

            with column:

                st.markdown(
                    f"### {item['emoji']}"
                )

                st.write(f"**{item['label']}**")

                st.caption(
                    f"{item['confidence'] * 100:.1f}% confidence"
                )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    else:
        st.caption("Your recent predictions will appear here.")

    st.divider()

    st.caption(
        "Note: This educational model is designed for cat and dog "
        "images. Other image categories may produce unreliable results."
    )


# ============================================
# ABOUT PROJECT PAGE
# ============================================

elif page == "📘 About Project":

    st.markdown(
        '<div class="hero-tag">ABOUT THE PROJECT</div>',
        unsafe_allow_html=True
    )

    st.title("Meet PawSense AI 🐾")

    st.markdown(
        "An intelligent image classification application "
        "built using deep learning and transfer learning."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Training Accuracy", "97.09%")

    with col2:
        st.metric("Validation Accuracy", "98.41%")

    with col3:
        st.metric("Image Categories", "02")

    st.divider()

    st.subheader("🧠 Technology Stack")

    technologies = [
        ("Python", "Programming language"),
        ("TensorFlow", "Deep learning framework"),
        ("MobileNetV2", "Pretrained CNN architecture"),
        ("Streamlit", "Web application framework"),
        ("HTML / CSS", "Frontend styling"),
        ("JavaScript", "Interactive copy functionality"),
    ]

    for technology, description in technologies:
        st.write(f"**{technology}** — {description}")

    st.divider()

    st.subheader("⚙️ How the application works")

    st.write("**01. Upload:** User uploads a cat or dog image.")

    st.write("**02. Preprocessing:** Image is resized to 160 × 160.")

    st.write("**03. Prediction:** MobileNetV2-based model analyses it.")

    st.write("**04. Result:** Application displays the predicted class.")

    st.write("**05. Confidence:** Model confidence score is displayed.")

    st.divider()

    st.subheader("🎯 Project Objective")

    st.write(
        "The objective is to develop an image classification "
        "system that distinguishes between cats and dogs using "
        "a trained deep learning model."
    )


# ============================================
# FOOTER
# ============================================

st.divider()

st.markdown(
    '<div class="footer-text">'
    'PAWSENSE AI • POWERED BY TENSORFLOW • DEEP LEARNING PROJECT'
    '</div>',
    unsafe_allow_html=True
)

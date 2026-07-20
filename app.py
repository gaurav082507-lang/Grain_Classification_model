"""
Grain Classification App
Model: GrainModel.keras
Built by Gaurav Gupta
Python 3.11
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Grain Classifier",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------
# CLASS NAMES (from your confusion matrix)
# ----------------------------------------------------------------------
CLASS_NAMES = [
    "chaana_dal",
    "chole",
    "harbara",
    "masur_dal",
    "matki",
    "moong",
    "peanut",
    "rice",
    "tur_dal",
    "wheat",
]

# Friendly display names + short descriptions for the "Supported Classes" section
CLASS_INFO = {
    "chaana_dal":  ("Chana Dal",  "Split chickpeas, husked and polished"),
    "chole":       ("Chole (Chickpeas)", "Whole brown/white chickpeas"),
    "harbara":     ("Harbara",    "Whole green/brown gram variant"),
    "masur_dal":   ("Masur Dal",  "Split red lentils"),
    "matki":       ("Matki",      "Moth beans, small brown grains"),
    "moong":       ("Moong",      "Whole green gram"),
    "peanut":      ("Peanut",     "Groundnut kernels"),
    "rice":        ("Rice",       "Polished white rice grains"),
    "tur_dal":     ("Tur Dal",    "Split pigeon peas"),
    "wheat":       ("Wheat",      "Whole wheat grains"),
}

IMG_SIZE = (224, 224)  # adjust to match your training input size
MODEL_PATH = "GrainModel.keras"

# ----------------------------------------------------------------------
# STYLING — grain-themed green gradient
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Grain-themed layered gradient background */
    .stApp {
        background:
            radial-gradient(circle at 15% 20%, rgba(255, 244, 214, 0.10) 0%, transparent 40%),
            radial-gradient(circle at 85% 80%, rgba(255, 244, 214, 0.08) 0%, transparent 45%),
            linear-gradient(160deg, #0b3d2e 0%, #145c44 35%, #1f7a5c 65%, #2f9e73 100%);
        background-attachment: fixed;
    }

    /* subtle wheat-grain texture using repeating dots/lines */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image:
            repeating-linear-gradient(115deg, rgba(255,236,179,0.035) 0px, rgba(255,236,179,0.035) 2px, transparent 2px, transparent 26px),
            repeating-linear-gradient(25deg, rgba(255,236,179,0.03) 0px, rgba(255,236,179,0.03) 2px, transparent 2px, transparent 30px);
        pointer-events: none;
        z-index: 0;
    }

    .block-container {
        padding-top: 2.2rem;
        max-width: 780px;
    }

    /* Header banner */
    .grain-header {
        text-align: center;
        padding: 1.4rem 1rem 1.6rem 1rem;
        margin-bottom: 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.03));
        border: 1px solid rgba(255, 236, 179, 0.25);
        backdrop-filter: blur(6px);
    }
    .grain-badge {
        display: inline-block;
        background: rgba(255, 236, 179, 0.15);
        border: 1px solid rgba(255, 236, 179, 0.35);
        color: #ffe9a8;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-bottom: 0.7rem;
    }
    .grain-title {
        font-family: 'Playfair Display', serif;
        font-weight: 800;
        font-size: 2.5rem;
        color: #fdf6e3;
        margin: 0.2rem 0 0.4rem 0;
        line-height: 1.15;
        text-shadow: 0 2px 14px rgba(0,0,0,0.25);
    }
    .grain-subtitle {
        color: #d7ecdf;
        font-size: 1.02rem;
        font-weight: 400;
        max-width: 520px;
        margin: 0 auto;
        line-height: 1.5;
    }
    .grain-stats {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 0.6rem 1.4rem;
        margin-top: 1rem;
        padding-top: 0.9rem;
        border-top: 1px solid rgba(255, 236, 179, 0.2);
    }
    .grain-stats span {
        color: #eaf6ef;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Glass card — styles Streamlit's native st.container(border=True) wrapper.
       (We use real containers instead of raw <div> tags because st.markdown
       calls don't nest around other widgets — a manual </div> hack renders
       as its own empty box rather than actually wrapping anything.) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.92) !important;
        border-radius: 18px !important;
        border: 1px solid rgba(255,255,255,0.4) !important;
        box-shadow: 0 12px 32px rgba(0,0,0,0.22);
        margin-bottom: 1.4rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border-radius: 18px !important;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #eafff3, #ffffff);
        border-left: 6px solid #1f7a5c;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-top: 1rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }
    .result-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #4a7c68;
        font-weight: 700;
    }
    .result-class {
        font-family: 'Playfair Display', serif;
        font-size: 2.1rem;
        font-weight: 800;
        color: #0b3d2e;
        margin: 0.2rem 0 0.6rem 0;
    }
    .result-conf {
        color: #2f2f2f;
        font-size: 0.95rem;
    }

    /* Class chips grid */
    .class-chip {
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(31, 122, 92, 0.25);
        border-radius: 12px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.6rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .class-chip:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.15);
    }
    .class-chip-name {
        font-weight: 700;
        color: #0b3d2e;
        font-size: 0.98rem;
    }
    .class-chip-desc {
        color: #5a6b62;
        font-size: 0.82rem;
        margin-top: 2px;
    }

    /* Footer */
    .grain-footer {
        text-align: center;
        margin-top: 2.2rem;
        padding: 1.1rem;
        color: #eaf6ef;
        font-size: 0.9rem;
    }
    .grain-footer a {
        color: #ffe9a8;
        font-weight: 700;
        text-decoration: none;
        border-bottom: 1px solid rgba(255,233,168,0.5);
        padding-bottom: 1px;
    }
    .grain-footer a:hover {
        border-bottom-color: #ffe9a8;
    }

    section[data-testid="stFileUploaderDropzone"] {
        background: #f7fbf8;
        border: 2px dashed #1f7a5c55;
        border-radius: 14px;
    }

    /* Main "Classify Grain" button — covers all Streamlit button class variants */
    .stButton>button,
    .stButton>button p,
    div[data-testid="stButton"] button,
    button[kind="primary"],
    button[kind="secondary"] {
        background: linear-gradient(135deg, #1f7a5c, #145c44) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 0.6rem 1.2rem !important;
        width: 100%;
        transition: filter 0.15s ease;
    }
    .stButton>button:hover {
        filter: brightness(1.15);
    }

    /* File uploader's internal "Browse files" button */
    section[data-testid="stFileUploaderDropzone"] button,
    section[data-testid="stFileUploaderDropzone"] button p {
        background: #ffffff !important;
        color: #145c44 !important;
        border: 1.5px solid #1f7a5c !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    section[data-testid="stFileUploaderDropzone"] small,
    section[data-testid="stFileUploaderDropzone"] span,
    section[data-testid="stFileUploaderDropzone"] div {
        color: #2f2f2f !important;
    }

    /* Uploaded file chip */
    div[data-testid="stFileUploaderFile"] {
        background: #ffffff !important;
        border-radius: 10px;
        border: 1px solid #1f7a5c33;
    }
    div[data-testid="stFileUploaderFile"] span,
    div[data-testid="stFileUploaderFile"] small {
        color: #2f2f2f !important;
    }

    /* st.info / st.error boxes -- force readable text */
    div[data-testid="stAlert"] {
        background: #ffffff !important;
        border-radius: 10px;
    }
    div[data-testid="stAlert"] p {
        color: #145c44 !important;
        font-weight: 500;
    }

    /* Custom probability bar rows (replaces st.progress) */
    .prob-row {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 0.55rem;
    }
    .prob-label {
        width: 105px;
        flex-shrink: 0;
        font-size: 0.9rem;
        font-weight: 600;
        color: #0b3d2e;
    }
    .prob-track {
        flex-grow: 1;
        height: 10px;
        background: #e4ede8;
        border-radius: 999px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #1f7a5c, #2f9e73);
    }
    .prob-pct {
        width: 60px;
        flex-shrink: 0;
        text-align: right;
        font-size: 0.85rem;
        font-weight: 600;
        color: #2f2f2f;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <div class="grain-header">
        <div class="grain-badge">🌾 EfficientNetB0 · {len(CLASS_NAMES)} Classes</div>
        <div class="grain-title">Grain &amp; Pulse Classifier</div>
        <div class="grain-subtitle">
            Upload a clear photo of grains or pulses and let the AI identify
            the type instantly — from dals and lentils to rice and wheat.
        </div>
        <div class="grain-stats">
            <span>🧠 EfficientNetB0 backbone</span>
            <span>📸 Trained on 7,500+ images</span>
            <span>🌾 {len(CLASS_NAMES)} grain &amp; pulse classes</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# MODEL LOADING (cached)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading GrainModel...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


# ----------------------------------------------------------------------
# UPLOAD + PREDICT CARD
# ----------------------------------------------------------------------
with st.container(border=True):
    uploaded_file = st.file_uploader(
        "Drop a grain image here, or browse to upload",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("🔍 Classify Grain"):
            try:
                model = load_model()
                processed = preprocess_image(image)
                preds = model.predict(processed)[0]

                top_idx = int(np.argmax(preds))
                top_class = CLASS_NAMES[top_idx]
                top_conf = float(preds[top_idx]) * 100
                display_name, display_desc = CLASS_INFO.get(top_class, (top_class, ""))

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Predicted Class</div>
                        <div class="result-class">🌾 {display_name}</div>
                        <div class="result-conf">Confidence: <b>{top_conf:.2f}%</b> &nbsp;·&nbsp; {display_desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("View full class probabilities", expanded=True):
                    sorted_idx = np.argsort(preds)[::-1]
                    rows_html = ""
                    for i in sorted_idx:
                        cname = CLASS_INFO.get(CLASS_NAMES[i], (CLASS_NAMES[i], ""))[0]
                        pct = preds[i] * 100
                        rows_html += f"""
                        <div class="prob-row">
                            <div class="prob-label">{cname}</div>
                            <div class="prob-track">
                                <div class="prob-fill" style="width:{pct:.2f}%;"></div>
                            </div>
                            <div class="prob-pct">{pct:.2f}%</div>
                        </div>
                        """
                    st.markdown(rows_html, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Model prediction failed: {e}")
                st.info(
                    f"Make sure **{MODEL_PATH}** is in the same directory as this app, "
                    "and that the input image size matches the model's expected input shape."
                )
    else:
        st.info("👆 Select an image to enable classification.")

# ----------------------------------------------------------------------
# SUPPORTED CLASSES SECTION
# ----------------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        f"### 🌱 Supported Grains &amp; Pulses <span style='font-size:0.9rem; font-weight:400; color:#5a6b62;'>({len(CLASS_NAMES)} classes)</span>",
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    for i, cname in enumerate(CLASS_NAMES):
        display_name, display_desc = CLASS_INFO.get(cname, (cname, ""))
        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="class-chip">
                    <div class="class-chip-name">🌾 {display_name}</div>
                    <div class="class-chip-desc">{display_desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ----------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="grain-footer">
        Model: <b>EfficientNetB0</b> · trained on <b>7,500+ images</b><br>
        Built by <b>Gaurav Gupta</b> &nbsp;·&nbsp;
        <a href="https://www.linkedin.com/in/gaurav-gupta-79754a377" target="_blank">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)

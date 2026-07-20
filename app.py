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
# TEST-SET CONFUSION MATRIX (from your evaluation run)
# true_class -> {predicted_class: count}. Only non-zero cells needed.
# Used to build a plain-language accuracy report for the UI.
# ----------------------------------------------------------------------
CONFUSION_DATA = {
    "chaana_dal": {"chaana_dal": 111, "tur_dal": 2},
    "chole":      {"chole": 113},
    "harbara":    {"harbara": 88, "chole": 20, "masur_dal": 5},
    "masur_dal":  {"chaana_dal": 10, "masur_dal": 103},
    "matki":      {"matki": 113},
    "moong":      {"moong": 113},
    "peanut":     {"chole": 111, "peanut": 2},
    "rice":       {"rice": 113},
    "tur_dal":    {"tur_dal": 113},
    "wheat":      {"masur_dal": 1, "wheat": 112},
}


def _class_accuracy_report():
    """Returns list of dicts: {class, total, correct, accuracy, top_confusion}."""
    report = []
    for cname, preds in CONFUSION_DATA.items():
        total = sum(preds.values())
        correct = preds.get(cname, 0)
        accuracy = (correct / total * 100) if total else 0.0
        confusions = {k: v for k, v in preds.items() if k != cname}
        top_confusion = max(confusions.items(), key=lambda x: x[1]) if confusions else None
        report.append({
            "class": cname,
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "top_confusion": top_confusion,  # (predicted_class, count) or None
        })
    return sorted(report, key=lambda r: r["accuracy"])
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
        background-color: rgba(255, 255, 255, 0.92) !important;
        border-radius: 18px !important;
        border: 1px solid rgba(255,255,255,0.4) !important;
        box-shadow: 0 12px 32px rgba(0,0,0,0.22);
        margin-bottom: 1.4rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div,
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] {
        background: transparent !important;
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

    /* Accuracy report rows */
    .acc-row {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 0.7rem;
    }
    .acc-label {
        width: 110px;
        flex-shrink: 0;
        font-size: 0.9rem;
        font-weight: 600;
        color: #0b3d2e !important;
    }
    .acc-track {
        flex-grow: 1;
        height: 14px;
        background: #e9eeec;
        border-radius: 999px;
        overflow: hidden;
    }
    .acc-fill-good { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #1f7a5c, #37b177); }
    .acc-fill-ok   { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #c98a1f, #e0a83e); }
    .acc-fill-bad  { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #b3402f, #d4573f); }
    .acc-pct {
        width: 58px;
        flex-shrink: 0;
        text-align: right;
        font-size: 0.85rem;
        font-weight: 700;
        color: #1a1a1a !important;
    }
    .acc-note {
        width: 100%;
        font-size: 0.78rem;
        color: #4a5a52 !important;
        margin: -0.3rem 0 0.9rem 0;
        padding-left: 118px;
    }

    /* Insight callout (flags real data/model issues in plain language) */
    .insight-callout {
        background: #fff4ec;
        border-left: 5px solid #c9622b;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin: 1rem 0;
    }
    .insight-callout .insight-title {
        font-weight: 700;
        color: #8a3f16 !important;
        font-size: 0.92rem;
        margin-bottom: 0.25rem;
    }
    .insight-callout .insight-body {
        color: #3a2a1c !important;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .accuracy-summary {
        text-align: center;
        padding: 0.6rem 0 1.2rem 0;
    }
    .accuracy-summary .big-num {
        font-family: 'Playfair Display', serif;
        font-size: 2.6rem;
        font-weight: 800;
        color: #0b3d2e !important;
    }
    .accuracy-summary .big-label {
        color: #4a5a52 !important;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
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
    """
    CONFIRMED from training notebook: images went straight from
    image_dataset_from_directory (raw 0-255 pixels, no rescale) through
    data_augmentation into EfficientNetB0(include_top=False) with no
    manual normalization step. Since TF 2.3+, EfficientNetB0 has its
    Rescaling + Normalization built into the model's own first layers,
    so raw 0-255 pixel input is correct here -- do not divide by 255
    or call efficientnet.preprocess_input (it's a no-op anyway).
    """
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")  # raw 0-255, no normalization
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
# MODEL ACCURACY REPORT (explainable version of the confusion matrix)
# ----------------------------------------------------------------------
with st.container(border=True):
    st.markdown("### 📊 How Accurate Is This Model?")
    st.caption(
        "Based on testing the model against 1,130 labeled sample images "
        "(113 per grain type) it hadn't seen during training."
    )

    _report = _class_accuracy_report()
    _total_correct = sum(r["correct"] for r in _report)
    _total_images = sum(r["total"] for r in _report)
    _overall_acc = _total_correct / _total_images * 100

    st.markdown(
        f"""
        <div class="accuracy-summary">
            <div class="big-num">{_overall_acc:.1f}%</div>
            <div class="big-label">Overall accuracy across all classes</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Per-class accuracy, worst first, so problems are visible immediately
    rows_html = ""
    notes_html = ""
    for r in _report:
        display_name = CLASS_INFO.get(r["class"], (r["class"], ""))[0]
        acc = r["accuracy"]
        tier = "good" if acc >= 95 else ("ok" if acc >= 80 else "bad")
        rows_html += f"""
        <div class="acc-row">
            <div class="acc-label">{display_name}</div>
            <div class="acc-track">
                <div class="acc-fill-{tier}" style="width:{acc:.1f}%;"></div>
            </div>
            <div class="acc-pct">{acc:.1f}%</div>
        </div>
        """
        if r["top_confusion"]:
            conf_class, conf_count = r["top_confusion"]
            conf_name = CLASS_INFO.get(conf_class, (conf_class, ""))[0]
            rows_html += (
                f'<div class="acc-note">Correctly identified {r["correct"]} of '
                f'{r["total"]} test images &nbsp;·&nbsp; most often mistaken for '
                f'<b>{conf_name}</b> ({conf_count} times)</div>'
            )
        else:
            rows_html += (
                f'<div class="acc-note">Correctly identified all {r["correct"]} '
                f'of {r["total"]} test images</div>'
            )

    st.markdown(rows_html, unsafe_allow_html=True)

    # Plain-language callout for the one class with a real, notable issue
    st.markdown(
        """
        <div class="insight-callout">
            <div class="insight-title">⚠️ Known issue: Peanut</div>
            <div class="insight-body">
                The model almost never recognizes <b>Peanut</b> correctly — it
                gets identified as <b>Chole (Chickpeas)</b> instead in the vast
                majority of test cases. This isn't normal model uncertainty;
                a mistake this one-sided and consistent usually means the
                training photos for Peanut and Chole were accidentally
                mixed up or mislabeled. If Peanut predictions matter to you,
                it's worth double-checking those two training folders before
                retraining.
            </div>
        </div>
        <div class="insight-callout" style="border-left-color:#1f7a5c; background:#eefaf3;">
            <div class="insight-title" style="color:#145c44;">ℹ️ Minor mix-up: Harbara &amp; Chole</div>
            <div class="insight-body">
                Harbara is correctly identified about 8 times out of 10. When
                it's wrong, it's almost always mistaken for Chole — likely
                because the two grains look visually similar (size, color,
                and shape). All other grain types in this model are
                identified correctly virtually every time.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

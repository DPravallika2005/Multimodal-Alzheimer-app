import streamlit as st
import torch
import pandas as pd
from utils import load_models
import base64
import os
import gdown  # pip install gdown

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Multimodal Alzheimer Detection",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Background image function
# -----------------------------
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Main heading */
        h1 {{
            color: #9400D3 !important;
        }}

        /* Subheaders */
        h2, h3 {{
            color: color-mix(in srgb, #EE82EE, #000 30%) !important;
        }}

        /* Number inputs */
        .stNumberInput input {{
            background-color: rgba(255,255,255,0.85);
            border-radius: 10px;
            padding: 5px;
        }}

        /* Text area */
        .stTextArea textarea {{
            background-color: rgba(255,255,255,0.85);
            border-radius: 10px;
            padding: 5px;
        }}

        /* File uploader */
        section[data-testid="stFileUploader"] {{
            border-radius: 10px;
            border: 2px dashed #E6D6FF;
            padding: 12px;
            background-color: rgba(255,255,255,0.25);
        }}

        /* Button */
        .stButton>button {{
            background-color: #A259FF;
            color: white;
            border-radius: 12px;
            height: 3em;
            width: 100%;
            font-size: 16px;
        }}

        /* Metric container */
        div[data-testid="stMetric"] {{
            background-color: rgba(255,255,255,0.9);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #FF0000;
        }}

        /* Metric Label */
        div[data-testid="stMetricLabel"] p {{
            color: black !important;
            font-weight: normal !important;
            font-size: 1.1rem !important;
        }}

        /* Metric Value */
        div[data-testid="stMetricValue"] > div {{
            color: black !important;
            font-weight: normal !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("med6.jpg")

# -----------------------------
# Main title
# -----------------------------
st.markdown(
    "<h1 style='text-align:center;'>🧠 Multimodal Alzheimer Detection System</h1>",
    unsafe_allow_html=True
)
st.divider()

# -----------------------------
# Load large BERT model dynamically (optional)
# -----------------------------
model_path = "models/Alzheimer_BERT_Model"
if not os.path.exists(model_path):
    url = "https://drive.google.com/drive/folders/1Fkrq5q8cfjZ1zzz5Zhg2mLCwqtWzzTQc"  # <-- Replace this
    gdown.download_folder(url, output=model_path, quiet=False)

# -----------------------------
# Load other models
# -----------------------------
resnet, densenet, mri_model, cognitive_model, tokenizer, speech_model = load_models()

# -----------------------------
# Centered input layout
# -----------------------------
left, center, right = st.columns([1,2,1])

with center:
    st.subheader("Patient Data Input")

    mmse = st.number_input("MMSE Score", min_value=0.0)
    adas = st.number_input("ADAS TOTAL13", min_value=0.0)
    faq = st.number_input("FAQTOTAL", min_value=0.0)

    speech_text = st.text_area("Speech Transcript")

    uploaded_image = st.file_uploader(
        "Upload MRI Image",
        type=["jpg","png","jpeg"]
    )

    st.markdown(" ")
    predict = st.button("Predict")

# -----------------------------
# Prediction logic
# -----------------------------
if predict:
    probs = {}

    # Cognitive model
    if mmse and adas and faq:
        df = pd.DataFrame(
            [[mmse, adas, faq]],
            columns=["MMSCORE", "TOTAL13", "FAQTOTAL"]
        )
        probs["cognitive"] = cognitive_model.predict_proba(df)[0][1]

    # Speech model
    if speech_text:
        inputs = tokenizer(
            speech_text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        with torch.no_grad():
            outputs = speech_model(**inputs)
        probs["speech"] = torch.softmax(outputs.logits, dim=1)[0][1].item()

    # Model weights
    weights = {"cognitive":0.3, "speech":0.3, "mri":0.4}
    active = {k: weights[k] for k in probs}
    w_sum = sum(active.values())
    normalized = {k: v / w_sum for k,v in active.items()}
    final_prob = sum(probs[k] * normalized[k] for k in probs)
    pred = "Alzheimer Detected" if final_prob > 0.5 else "Normal"

    st.divider()
    r1, r2, r3 = st.columns([1,2,1])
    with r2:
        st.subheader("Prediction Result")
        st.metric("Alzheimer Probability", f"{final_prob:.3f}")
        if final_prob > 0.5:
            st.error("⚠️ Alzheimer Detected")
        else:
            st.success("✅ Normal")
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "final_model_bundle.joblib"

st.set_page_config(page_title="AI Smart Healthcare", page_icon="🩺", layout="wide")
st.title("🩺 AI in Smart Healthcare for Early Diagnosis")
st.caption("Explainable, calibration-aware research prototype")
st.warning(
    "Research prototype only. This estimates risk from the study dataset and is not a medical diagnosis."
)

if not MODEL_PATH.exists():
    st.error("Model not found. Run: python -m src.train")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["raw_feature_names"]

st.subheader("Questionnaire / patient information")
inputs = {}
for feature in features:
    label = feature.replace("_", " ").title()
    if feature == "age":
        inputs[feature] = st.number_input(label, min_value=1, max_value=120, value=40, step=1)
    else:
        inputs[feature] = st.selectbox(label, ["No", "Yes"], key=feature)

if st.button("Analyze Risk", type="primary"):
    row = pd.DataFrame([inputs])
    probability = float(model.predict_proba(row)[0, 1])
    output = "Elevated risk" if probability >= bundle["decision_threshold"] else "Lower predicted risk"
    st.metric("Estimated model probability", f"{probability * 100:.1f}%")
    st.write(f"**Model output:** {output}")
    st.info(
        "This is a model output learned from a public research dataset. "
        "It is not a clinical diagnosis and may not generalize to another population."
    )

st.divider()
st.caption("Dataset: UCI Early Stage Diabetes Risk Prediction.")

"""
Breast Cancer Detection — Streamlit app.

Loads the pre-trained pipeline (model.pkl) plus label_encoder.pkl and
feature_columns.pkl (produced by train_model.py), shows an input form for
the 15 tumor features, and predicts Benign vs Malignant.
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🎗️", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, label_encoder, feature_columns


model, label_encoder, feature_columns = load_artifacts()

st.title("🎗️ Breast Cancer Detection")
st.write(
    "Enter the tumor's measured and engineered features below. "
    "The model predicts whether the tumor is **Benign (B)** or **Malignant (M)**."
)

# (min, max, default) for each feature, based on the training dataset's
# describe() statistics — used to give sensible slider/number-input ranges.
FEATURE_RANGES = {
    "radius_mean": (5.0, 30.0, 14.10, 0.01),
    "texture_mean": (5.0, 45.0, 19.32, 0.01),
    "perimeter_mean": (30.0, 200.0, 91.75, 0.01),
    "area_mean": (100.0, 2600.0, 651.33, 1.0),
    "smoothness_mean": (0.03, 0.20, 0.0960, 0.0001),
    "compactness_mean": (0.00, 0.40, 0.1031, 0.0001),
    "concavity_mean": (-0.02, 0.50, 0.0867, 0.0001),
    "concave points_mean": (-0.01, 0.25, 0.0482, 0.0001),
    "shape_irregularity": (0.00, 1.00, 0.2379, 0.0001),
    "border_complexity": (-0.01, 0.10, 0.0070, 0.0001),
    "tumor_aggressiveness": (0.00, 0.35, 0.0793, 0.0001),
    "radius_texture_interaction": (50.0, 800.0, 276.77, 0.1),
    "radius_concavity_interaction": (-0.10, 12.0, 1.4085, 0.001),
    "concavity_density": (-0.0005, 0.0020, 0.000129, 0.000001),
    "malignancy_risk_score": (5.0, 75.0, 29.83, 0.01),
}

inputs = {}
with st.form("prediction_form"):
    cols = st.columns(2)
    for i, feature in enumerate(feature_columns):
        lo, hi, default, step = FEATURE_RANGES.get(feature, (0.0, 1.0, 0.5, 0.01))
        with cols[i % 2]:
            inputs[feature] = st.number_input(
                feature.replace("_", " ").title(),
                min_value=float(lo),
                max_value=float(hi),
                value=float(default),
                step=float(step),
                format="%.6f",
            )
    submitted = st.form_submit_button("Predict")

if submitted:
    X_input = pd.DataFrame([inputs])[feature_columns]

    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]
    label = label_encoder.inverse_transform([pred])[0]

    confidence = proba[pred] * 100

    if label == "M":
        st.error(f"🔴 Prediction: **Malignant**  (confidence: {confidence:.2f}%)")
    else:
        st.success(f"🟢 Prediction: **Benign**  (confidence: {confidence:.2f}%)")

    st.subheader("Prediction probabilities")
    prob_df = pd.DataFrame(
        {
            "Diagnosis": label_encoder.inverse_transform([0, 1]),
            "Probability": proba,
        }
    ).set_index("Diagnosis")
    st.bar_chart(prob_df)

    with st.expander("Show input values used"):
        st.dataframe(X_input.T.rename(columns={0: "value"}))

st.caption(
    "Model trained on the Breast Cancer Enhanced Dataset "
    "(5,500 samples, 15 features). For educational/demo purposes only — "
    "not a substitute for professional medical diagnosis."
)

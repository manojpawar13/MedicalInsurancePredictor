
from pathlib import Path

import streamlit as st

from src.preprocessing import make_input_frame
from src.prediction import load_artifacts, predict_charge


st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="🏥",
    layout="centered"
)


@st.cache_resource
def get_artifacts():
    return load_artifacts()


st.title("Medical Insurance Cost Prediction")

st.write(
    "Enter your details below to estimate medical insurance charges."
)

try:
    model, preprocessor = get_artifacts()
except Exception as exc:
    st.error("Unable to load the model files.")
    st.exception(exc)
    st.stop()


with st.form("insurance_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

        sex = st.selectbox(
            "Sex",
            ["female", "male"]
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=30.0,
            step=0.1
        )

    with col2:
        children = st.number_input(
            "Number of children",
            min_value=0,
            max_value=10,
            value=0
        )

        smoker = st.selectbox(
            "Smoker",
            ["no", "yes"]
        )

        region = st.selectbox(
            "Region",
            ["northeast", "northwest", "southeast", "southwest"]
        )

    submitted = st.form_submit_button(
        "Predict Insurance Charges",
        type="primary"
    )


if submitted:

    input_frame = make_input_frame(
        age,
        sex,
        bmi,
        children,
        smoker,
        region
    )

    prediction = predict_charge(
        input_frame,
        model,
        preprocessor
    )

    st.success(
        f"Estimated Insurance Charges: {prediction:,.2f}"
    )

    st.caption(
        "This is an educational model estimate, not an actual insurance quote."
    )

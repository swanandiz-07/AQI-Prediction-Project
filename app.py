import os
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌫️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")

EXPECTED_FEATURES = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"
]

AQI_CATEGORIES = [
    (50, "Good"),
    (100, "Satisfactory"),
    (200, "Moderate"),
    (300, "Poor"),
    (400, "Very Poor"),
    (500, "Severe"),
]


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if os.path.getsize(MODEL_PATH) == 0:
        raise ValueError(
            "best_model.joblib is empty. "
            "Replace it with the trained model bundle."
        )

    bundle = joblib.load(MODEL_PATH)

    required_keys = {"model", "features", "model_name"}
    if not isinstance(bundle, dict):
        raise ValueError(
            "The model file must contain the model bundle "
            "created by the training notebook."
        )

    if not required_keys.issubset(bundle.keys()):
        raise ValueError(
            f"Model bundle must contain: {sorted(required_keys)}"
        )

    if bundle["features"] != EXPECTED_FEATURES:
        raise ValueError(
            "Feature names or order do not match the app. "
            "Align EXPECTED_FEATURES with the training notebook."
        )

    return bundle


def get_aqi_category(aqi):
    for upper_limit, category in AQI_CATEGORIES:
        if aqi <= upper_limit:
            return category
    return None


# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🌫️ AQI Prediction System")
st.write(
    "Predict the Air Quality Index (AQI) from pollutant "
    "concentration measurements using a trained machine "
    "learning regression model."
)

st.info(
    "Enter pollutant concentrations in the same units and "
    "measurement format used in the training dataset."
)

# --------------------------------------------------
# Load model
# --------------------------------------------------
try:
    bundle = load_model()
    model = bundle["model"]
    features = bundle["features"]
    model_name = bundle["model_name"]

except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()

st.caption(f"Selected model: {model_name}")

# --------------------------------------------------
# Pollutant input form
# --------------------------------------------------
st.subheader("Enter pollutant measurements")

with st.form("aqi_prediction_form"):
    col1, col2, col3 = st.columns(3)
    inputs = {}

    for index, feature in enumerate(features):
        with [col1, col2, col3][index % 3]:
            inputs[feature] = st.number_input(
                feature,
                min_value=0.0,
                value=0.0,
                step=0.1,
                format="%.3f",
                key=f"input_{feature}"
            )

    submitted = st.form_submit_button(
        "Predict AQI",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if submitted:
    input_df = pd.DataFrame(
        [[inputs[feature] for feature in features]],
        columns=features
    )

    try:
        prediction = float(model.predict(input_df)[0])

        if not pd.notna(prediction):
            st.error("The model returned an invalid prediction.")
        else:
            st.subheader("Prediction result")

            st.metric(
                label="Predicted AQI",
                value=f"{prediction:.2f}"
            )

            if 0 <= prediction <= 500:
                category = get_aqi_category(prediction)
                st.success(f"AQI Category: {category}")
            else:
                st.warning(
                    "The model predicted a value outside 0–500. "
                    "The result has not been clipped. Check the "
                    "input measurements and training data."
                )

            with st.expander("Review submitted measurements"):
                st.dataframe(input_df, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")

# --------------------------------------------------
# Project information
# --------------------------------------------------
st.divider()
st.subheader("About this project")

st.write(
    "This project uses historical air-quality observations "
    "from India to train and compare regression models for "
    "AQI prediction."
)

st.write(
    "Algorithms compared: Linear Regression, Decision Tree "
    "Regressor, Random Forest Regressor, and Gradient "
    "Boosting Regressor."
)

st.caption(
    "This is an educational machine learning project. "
    "Predictions are estimates and should not replace official "
    "air-quality measurements or public-health guidance."
)


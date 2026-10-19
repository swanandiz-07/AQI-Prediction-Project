from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.impute import SimpleImputer

from src.preprocessing import (
    EXPECTED_FEATURES,
    build_input_frame,
    get_aqi_category,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
DATA_PATH = ROOT / "data" / "city_day_cleaned.csv"

METRICS = pd.DataFrame([
    {"Model": "Random Forest", "MAE": 20.7611, "RMSE": 40.6356, "R2": 0.9098},
    {"Model": "Gradient Boosting", "MAE": 23.6586, "RMSE": 43.3544, "R2": 0.8974},
    {"Model": "Decision Tree", "MAE": 24.7162, "RMSE": 50.8913, "R2": 0.8586},
    {"Model": "Linear Regression", "MAE": 31.1698, "RMSE": 59.4397, "R2": 0.8071},
])

st.set_page_config(
    page_title="AQI Dashboard",
    page_icon="🌿",
    layout="wide",
)

st.markdown("""
<style>
.block-container {max-width: 1450px; padding-top: 1.5rem;}
.hero {
    padding: 1.6rem 1.8rem;
    border-radius: 18px;
    color: white;
    background: linear-gradient(115deg, #052e2b, #0f766e, #0e7490);
    margin-bottom: 1rem;
}
.hero h1 {color: white; margin-bottom: .3rem;}
.hero p {color: #e0f2f1; margin-bottom: 0;}
[data-testid="stMetric"] {
    padding: 12px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,.25);
}
</style>
<div class="hero">
<h1>🌿 AQI Dashboard</h1>
<p>Air quality across Indian cities, 2015–2020.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "AQI" in df.columns:
        df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
    return df


def fix_imputers(obj, seen=None):
    if seen is None:
        seen = set()
    if id(obj) in seen or obj is None:
        return
    seen.add(id(obj))

    if isinstance(obj, SimpleImputer):
        if not hasattr(obj, "_fill_dtype") and hasattr(obj, "statistics_"):
            obj._fill_dtype = np.asarray(obj.statistics_).dtype

    if isinstance(obj, (list, tuple)):
        for item in obj:
            fix_imputers(item, seen)
        return

    if isinstance(obj, dict):
        for item in obj.values():
            fix_imputers(item, seen)
        return

    for attr in (
        "steps", "transformers", "transformers_", "estimators",
        "estimators_", "estimator", "estimator_", "named_steps",
        "named_transformers_", "transformer_list",
    ):
        child = getattr(obj, attr, None)
        if child is not None and not isinstance(child, (str, int, float)):
            if hasattr(child, "items") and not isinstance(child, dict):
                child = dict(child.items())
            fix_imputers(child, seen)


@st.cache_resource
def load_model(path):
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError("The model file is empty.")

    bundle = joblib.load(path)
    required = {"model", "features", "model_name"}

    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError(
            "Expected a bundle dictionary with model, features, and model_name."
        )

    if list(bundle["features"]) != EXPECTED_FEATURES:
        raise ValueError("Feature names or order do not match the 12 expected features.")

    fix_imputers(bundle["model"])
    return bundle


@st.cache_data(show_spinner="Running the model on the dataset")
def evaluate(_model, features, df):
    cols = list(features) + ["AQI"]
    sub = df.dropna(subset=["AQI"]).copy()
    sub[list(features)] = sub[list(features)].apply(pd.to_numeric, errors="coerce")
    sub = sub.dropna(subset=cols)
    pred = _model.predict(sub[list(features)])
    out = pd.DataFrame({
        "Actual": sub["AQI"].to_numpy(),
        "Predicted": np.asarray(pred, dtype=float),
    })
    if "City" in sub.columns:
        out["City"] = sub["City"].astype(str).to_numpy()
    out["Residual"] = out["Actual"] - out["Predicted"]
    return out


def show_result(prediction, input_df, recorded=None):
    category = get_aqi_category(prediction)

    left, right = st.columns([1, 2])
    with left:
        st.metric("Predicted AQI", f"{prediction:.2f}")
        if recorded is not None:
            st.metric("Recorded AQI", f"{recorded:.2f}")
            st.metric("Difference", f"{prediction - recorded:+.2f}")
        st.subheader(f"Category: {category}")

    with right:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={"valueformat": ".2f"},
            title={"text": "Predicted AQI"},
            gauge={
                "axis": {"range": [0, 500]},
                "bar": {"color": "#0f766e"},
                "steps": [
                    {"range": [0, 50], "color": "#dcfce7"},
                    {"range": [50, 100], "color": "#ecfccb"},
                    {"range": [100, 200], "color": "#fef9c3"},
                    {"range": [200, 300], "color": "#ffedd5"},
                    {"range": [300, 400], "color": "#fee2e2"},
                    {"range": [400, 500], "color": "#fecaca"},
                ],
            },
        ))
        gauge.update_layout(height=280, margin=dict(l=15, r=15, t=45, b=5))
        st.plotly_chart(gauge, width="stretch")

    if prediction < 0 or prediction > 500:
        st.warning("The prediction is outside the 0–500 range.")

    with st.expander("Input values"):
        st.dataframe(input_df, width="stretch")


with st.sidebar:
    st.title("Navigation")
    page = st.radio(
        "Go to",
        [
            "Overview",
            "Historical Analysis",
            "Predict AQI",
            "Model Performance",
        ],
    )
    st.divider()
    st.caption("Dataset: Air Quality Data in India (2015–2020)")

data = None
if DATA_PATH.exists():
    try:
        data = load_data(str(DATA_PATH))
    except Exception as exc:
        st.sidebar.error(f"Could not read CSV: {exc}")


if page == "Overview":
    st.subheader("Overview")
    st.write(
        "Browse historical AQI data, compare the trained models, "
        "and predict AQI from pollutant values."
    )

    if data is None:
        st.warning(
            "Dataset not found. Place the cleaned CSV at "
            "`data/city_day_cleaned.csv`."
        )
    else:
        aq = data["AQI"].dropna() if "AQI" in data else pd.Series(dtype=float)
        cities = data["City"].nunique() if "City" in data else None

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records", f"{len(data):,}")
        c2.metric("Cities", f"{cities:,}" if cities is not None else "N/A")
        c3.metric("Records with AQI", f"{len(aq):,}")
        c4.metric("Mean AQI", f"{aq.mean():.1f}" if len(aq) else "N/A")

        if "Date" in data and data["Date"].notna().any() and len(aq):
            trend_data = data.copy()
            trend_data["Date"] = pd.to_datetime(trend_data["Date"], errors="coerce")
            trend_data["AQI"] = pd.to_numeric(trend_data["AQI"], errors="coerce")
            trend_data = trend_data.dropna(subset=["Date", "AQI"])

            monthly = (
                trend_data
                .set_index("Date")["AQI"]
                .resample("MS")
                .mean()
                .reset_index()
            )

            if not monthly.empty:
                fig = px.line(
                    monthly,
                    x="Date",
                    y="AQI",
                    markers=True,
                    title="Monthly Mean AQI",
                    labels={"Date": "Month", "AQI": "Average AQI"},
                )
                fig.update_layout(template="plotly_dark", hovermode="x unified")
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No valid date and AQI records for this chart.")

    st.markdown("### Model results")
    best = METRICS.sort_values("R2", ascending=False).iloc[0]
    a, b, c = st.columns(3)
    a.metric("Best model (R²)", best["Model"])
    b.metric("R²", f'{best["R2"]:.4f}')
    c.metric("RMSE", f'{best["RMSE"]:.2f}')


elif page == "Historical Analysis":
    st.subheader("Historical AQI analysis")

    if data is None:
        st.warning("Add `city_day_cleaned.csv` to the data folder to use this page.")
        st.stop()

    if "AQI" not in data.columns:
        st.error("The CSV must contain an AQI column.")
        st.stop()

    df = data.dropna(subset=["AQI"]).copy()

    left, right = st.columns(2)
    if "City" in df.columns:
        city_list = sorted(df["City"].dropna().astype(str).unique())
        selected_cities = left.multiselect(
            "Cities",
            city_list,
            default=city_list[:min(5, len(city_list))],
        )
        if selected_cities:
            df = df[df["City"].astype(str).isin(selected_cities)]

    if "Date" in df.columns and df["Date"].notna().any():
        min_date = df["Date"].min().date()
        max_date = df["Date"].max().date()
        date_range = right.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = date_range
            df = df[
                (df["Date"] >= pd.Timestamp(start))
                & (df["Date"] < pd.Timestamp(end) + pd.Timedelta(days=1))
            ]

    a, b, c = st.columns(3)
    a.metric("Observations", f"{len(df):,}")
    b.metric("Mean AQI", f'{df["AQI"].mean():.1f}' if len(df) else "N/A")
    c.metric("Median AQI", f'{df["AQI"].median():.1f}' if len(df) else "N/A")

    if df.empty:
        st.info("No records match the selected filters.")
        st.stop()

    if "Date" in df.columns and df["Date"].notna().any():
        if "City" in df.columns:
            monthly = df.dropna(subset=["Date"]).groupby(
                [pd.Grouper(key="Date", freq="MS"), "City"],
                as_index=False,
            )["AQI"].mean()
            fig = px.line(
                monthly, x="Date", y="AQI", color="City",
                title="Monthly Mean AQI by City",
            )
        else:
            monthly = df.dropna(subset=["Date"]).groupby(
                pd.Grouper(key="Date", freq="MS"), as_index=False
            )["AQI"].mean()
            fig = px.line(
                monthly, x="Date", y="AQI", markers=True,
                title="Monthly Mean AQI",
            )
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.histogram(df, x="AQI", nbins=40, title="AQI Distribution"),
            width="stretch",
        )

    if "City" in df.columns:
        city_mean = (
            df.groupby("City", as_index=False)["AQI"]
            .mean()
            .sort_values("AQI", ascending=False)
            .head(15)
        )
        with right:
            st.plotly_chart(
                px.bar(
                    city_mean.sort_values("AQI"),
                    x="AQI",
                    y="City",
                    orientation="h",
                    title="Cities with Highest Mean AQI",
                ),
                width="stretch",
            )

    pollutants = [name for name in EXPECTED_FEATURES if name in df.columns]
    if len(pollutants) >= 2:
        st.subheader("Pollutant correlation")
        corr_data = df[pollutants + ["AQI"]].apply(pd.to_numeric, errors="coerce")
        corr = corr_data.corr()
        st.plotly_chart(
            px.imshow(
                corr,
                text_auto=".2f",
                zmin=-1,
                zmax=1,
                color_continuous_scale="RdBu_r",
                title="Correlation Matrix",
            ),
            width="stretch",
        )

    with st.expander("Filtered records"):
        st.dataframe(df, width="stretch")


elif page == "Predict AQI":
    st.subheader("Predict AQI")

    try:
        bundle = load_model(MODEL_PATH)
    except Exception as exc:
        st.error(f"Could not load the model: {exc}")
        st.stop()

    model = bundle["model"]
    features = list(bundle["features"])
    st.caption(f"Model: {bundle['model_name']}")

    defaults = {
        "PM2.5": 25, "PM10": 50, "NO": 10, "NO2": 20,
        "NOx": 30, "NH3": 10, "CO": 1, "SO2": 10,
        "O3": 30, "Benzene": 1, "Toluene": 5, "Xylene": 1,
    }

    tab_manual, tab_lookup = st.tabs(["Enter values", "Pick a city and date"])

    with tab_manual:
        st.write("Enter pollutant values in the same units used for training.")

        with st.form("prediction_form"):
            columns = st.columns(3)
            values = {}

            for i, feature in enumerate(features):
                with columns[i % 3]:
                    values[feature] = st.number_input(
                        feature,
                        min_value=0.0,
                        value=float(defaults.get(feature, 0)),
                        step=0.1,
                        format="%.3f",
                    )

            submitted = st.form_submit_button(
                "Predict AQI",
                type="primary",
                width="stretch",
            )

        if submitted:
            try:
                input_df = build_input_frame(values, features)
                prediction = float(model.predict(input_df)[0])
                show_result(prediction, input_df)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    with tab_lookup:
        needed = set(features) | {"City", "Date", "AQI"}
        if data is None or not needed.issubset(data.columns):
            st.info(
                "This needs the dataset with City, Date, AQI and all pollutant columns."
            )
        else:
            lookup = data.dropna(subset=["City", "Date"]).copy()
            lookup["City"] = lookup["City"].astype(str)

            c1, c2 = st.columns(2)
            city = c1.selectbox("City", sorted(lookup["City"].unique()))
            city_rows = lookup[lookup["City"] == city].sort_values("Date")
            dates = city_rows["Date"].dt.date.unique().tolist()
            chosen_date = c2.selectbox("Date", dates)

            if st.button("Predict for this date", type="primary", width="stretch"):
                try:
                    row = city_rows[city_rows["Date"].dt.date == chosen_date].iloc[0]
                    row_values = {
                        f: pd.to_numeric(row[f], errors="coerce") for f in features
                    }
                    input_df = pd.DataFrame([row_values], columns=features)
                    prediction = float(model.predict(input_df)[0])
                    recorded = row["AQI"]
                    recorded = None if pd.isna(recorded) else float(recorded)
                    show_result(prediction, input_df, recorded)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")


elif page == "Model Performance":
    st.subheader("Model comparison")

    st.dataframe(
        METRICS.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}"}),
        width="stretch",
        hide_index=True,
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.bar(
                METRICS.sort_values("RMSE"),
                x="Model",
                y="RMSE",
                text_auto=".2f",
                title="RMSE (lower is better)",
            ),
            width="stretch",
        )

    with right:
        fig = px.bar(
            METRICS.sort_values("R2", ascending=False),
            x="Model",
            y="R2",
            text_auto=".3f",
            title="R² (higher is better)",
        )
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, width="stretch")

    st.markdown("""
- **MAE:** average absolute error in AQI points.
- **RMSE:** like MAE but penalizes large errors more.
- **R²:** share of AQI variation the model explains on the test set.
""")

    st.subheader("Actual vs predicted")

    try:
        bundle = load_model(MODEL_PATH)
    except Exception as exc:
        st.error(f"Could not load the model: {exc}")
        st.stop()

    needed = set(bundle["features"]) | {"AQI"}
    if data is None or not needed.issubset(data.columns):
        st.info("The dataset with all pollutant columns and AQI is needed for these plots.")
        st.stop()

    try:
        result = evaluate(bundle["model"], tuple(bundle["features"]), data)
    except Exception as exc:
        st.error(f"Could not evaluate the model: {exc}")
        st.stop()

    if result.empty:
        st.info("No complete rows are available for evaluation.")
        st.stop()

    mae = result["Residual"].abs().mean()
    rmse = float(np.sqrt((result["Residual"] ** 2).mean()))
    ss_res = float((result["Residual"] ** 2).sum())
    ss_tot = float(((result["Actual"] - result["Actual"].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows used", f"{len(result):,}")
    m2.metric("MAE", f"{mae:.2f}")
    m3.metric("RMSE", f"{rmse:.2f}")
    m4.metric("R²", f"{r2:.4f}")

    plot_df = result.sample(min(len(result), 4000), random_state=42)
    axis_max = float(max(plot_df["Actual"].max(), plot_df["Predicted"].max()))

    left, right = st.columns(2)
    with left:
        scatter = px.scatter(
            plot_df,
            x="Actual",
            y="Predicted",
            opacity=0.45,
            hover_data=["City"] if "City" in plot_df.columns else None,
            title="Actual vs Predicted AQI",
        )
        scatter.add_trace(go.Scatter(
            x=[0, axis_max],
            y=[0, axis_max],
            mode="lines",
            name="Perfect prediction",
            line=dict(color="#ef4444", dash="dash"),
        ))
        st.plotly_chart(scatter, width="stretch")

    with right:
        resid = px.scatter(
            plot_df,
            x="Predicted",
            y="Residual",
            opacity=0.45,
            hover_data=["City"] if "City" in plot_df.columns else None,
            title="Residuals vs Predicted AQI",
            labels={"Residual": "Residual (actual − predicted)"},
        )
        resid.add_hline(y=0, line_dash="dash", line_color="#ef4444")
        st.plotly_chart(resid, width="stretch")

    st.plotly_chart(
        px.histogram(
            result,
            x="Residual",
            nbins=60,
            title="Residual Distribution",
        ),
        width="stretch",
    )
    st.caption(
        "These plots use every complete row in the dataset, including rows the "
        "model was trained on, so the fit looks tighter than the test-set metrics above."
    )
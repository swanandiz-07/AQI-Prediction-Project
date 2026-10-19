
import math
import pandas as pd

EXPECTED_FEATURES = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"
]


def build_input_frame(values, features=None):
    """Validate pollutant values and preserve trained feature order."""
    order = list(features or EXPECTED_FEATURES)
    missing = [name for name in order if name not in values]

    if missing:
        raise ValueError(
            "Missing pollutant inputs: " + ", ".join(missing)
        )

    row = {}
    for name in order:
        try:
            value = float(values[name])
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be numeric.") from None

        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"{name} must be a finite, non-negative number."
            )

        row[name] = value

    return pd.DataFrame([row], columns=order)


def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    if aqi <= 500:
        return "Severe"
    return "Unclassified"

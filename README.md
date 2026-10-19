# AQI Prediction Using Machine Learning

An interactive dashboard for exploring historical air quality data in India and estimating the Air Quality Index (AQI) using regression models.

## Project Overview

Air pollution affects public health and environmental quality. This project uses historical air quality observations to investigate pollutant patterns, compare machine learning regression models, and estimate AQI from pollutant concentration measurements.

## Objectives

- Clean and analyze historical air quality data.
- Explore AQI trends, pollutant distributions, and relationships between pollutants.
- Train and compare multiple regression algorithms.
- Evaluate model performance using MAE, MSE, RMSE, and R².
- Develop an interactive dashboard for AQI estimation and data visualization.

## Dataset

**Dataset:** Air Quality Data in India (2015–2020)  
**Source:** [Kaggle — Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

The original `city_day.csv` dataset contains daily air quality observations for Indian cities, including pollutant measurements, dates, and AQI values.

The cleaned dataset used by the dashboard is expected at:

`data/city_day_cleaned.csv`

## Features Used

The model uses these 12 pollutant features:

- PM2.5
- PM10
- NO
- NO2
- NOx
- NH3
- CO
- SO2
- O3
- Benzene
- Toluene
- Xylene

## Machine Learning Models

The project compares:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

The preferred model is selected using evaluation results on held-out test data rather than assuming that one algorithm will always perform best.

## Evaluation Metrics

- **MAE:** Mean Absolute Error; average absolute difference between actual and predicted AQI.
- **MSE:** Mean Squared Error; penalizes larger errors more strongly.
- **RMSE:** Root Mean Squared Error; expresses prediction error on the target's numerical scale.
- **R²:** Coefficient of determination; describes the proportion of target variance explained by the model.

Final evaluation values should be taken from the final model-training notebook.

## Dashboard

The Streamlit dashboard is designed to provide:

- **Overview:** Dataset statistics and historical AQI summaries.
- **Historical Analysis:** AQI trends, AQI distribution, city comparisons, and pollutant correlations when the cleaned dataset is available.
- **Predict AQI:** A form for the 12 pollutant inputs, a predicted AQI value, and an AQI category.
- **Model Performance:** A comparison of regression metrics.

Historical visualizations are calculated from the cleaned dataset. Saved graphs from the analysis notebook may also be retained as project figures.

## Technology Stack

- Python
- Google Colab
- Pandas and NumPy
- Matplotlib and Plotly
- Scikit-learn
- Joblib
- Streamlit
- Git and GitHub

## Run Globally 

### Streamlit URL : [AQI-Prediction-Project](https://aqi-prediction-project-eujzeu9nybedsljqee9zix.streamlit.app/)

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/swanandiz-07/AQI-Prediction-Project.git
cd AQI-Prediction-Project
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Add the required files

Place the cleaned CSV at `data/city_day_cleaned.csv`.

Place the trained model bundle at `models/best_model.joblib`. The bundle must contain the trained estimator or pipeline, its feature list, and the model name.

Use the same scikit-learn version used to serialize the model.

### 5. Launch the dashboard

```powershell
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually `http://localhost:8501`.

### 6. Run automated tests

```powershell
python -m unittest discover -s tests -v
```

## Limitations

- Predictions depend on the quality and coverage of the historical data.
- Missing measurements, outliers, and differences between cities can affect model performance.
- The dashboard's estimates are not official air quality measurements.
- Estimating AQI from supplied pollutant measurements is not the same as forecasting future AQI.



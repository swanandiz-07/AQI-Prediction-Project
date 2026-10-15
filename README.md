# Air Quality Index (AQI) Prediction Using Machine Learning

## About the Project

Air pollution is a major environmental concern, especially in urban areas where pollutant levels can change depending on traffic, industrial activity, weather conditions, and other factors. Monitoring air quality helps us understand pollution levels and the potential risks associated with them.

This project aims to develop a machine learning model that predicts the Air Quality Index (AQI) using pollutant concentration measurements. We will compare different regression algorithms to identify a model that performs well on unseen data and build an interactive dashboard using Streamlit to make predictions easier to access.

The project is being developed as part of our machine learning coursework by a team of three members.

## Objectives

- Study and preprocess historical air quality data.
- Handle missing values, duplicate records, and potential outliers.
- Explore pollutant distributions and their relationships with AQI.
- Train and compare multiple machine learning regression models.
- Evaluate model performance using standard regression metrics.
- Develop a Streamlit dashboard for AQI prediction and data visualization.
- Document the complete workflow and make the project reproducible through GitHub.

## Dataset

**Dataset:** Air Quality Data in India (2015–2020)

**Source:** [Kaggle – Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

The project uses the `city_day.csv` file, which contains daily air quality observations for multiple Indian cities. It includes pollutant measurements, dates, city information, AQI values, and AQI categories.

The dataset contains variables such as:

- PM2.5
- PM10
- Nitrogen dioxide (NO2)
- Sulphur dioxide (SO2)
- Carbon monoxide (CO)
- Ozone (O3)
- AQI
- AQI category
- City and date

The dataset also includes other pollutant measurements that may be considered during exploratory analysis.

**Note:** The standard `city_day.csv` file does not include temperature, humidity, or wind speed. The initial version of this project therefore focuses on the available pollutant measurements. Weather variables may be included in a later version if suitable historical weather data can be obtained and matched correctly with the pollution records.

## Project Workflow

The project follows these main stages:

1. **Data collection:** Download the historical air quality dataset from Kaggle.
2. **Data cleaning:** Inspect data types, missing values, duplicates, invalid measurements, and potential outliers.
3. **Exploratory data analysis:** Study pollutant distributions, AQI trends, city-wise differences, and correlations.
4. **Feature preparation:** Select suitable pollutant measurements as input features and AQI as the regression target.
5. **Model training:** Train and compare multiple regression algorithms using consistent training and testing data.
6. **Model evaluation:** Compare prediction errors and goodness of fit using standard metrics.
7. **Dashboard development:** Build an interactive Streamlit interface for AQI prediction and visual analysis.
8. **Testing and deployment:** Test the application, document setup instructions, and make it accessible through a web link if deployment is successful.

## Machine Learning Models

The following regression models are planned for comparison:

| Model | Purpose |
|---|---|
| Linear Regression | Baseline model for understanding linear relationships |
| Random Forest Regressor | Captures nonlinear relationships using multiple decision trees |
| Gradient Boosting Regressor | Builds an ensemble of trees sequentially to reduce prediction errors |
| XGBoost Regressor | Uses gradient-boosted trees for efficient predictive modelling |

The final model will be selected based on its evaluation results rather than assuming that one algorithm will always perform best.

## Evaluation Metrics

The models will be evaluated using:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted AQI.
- **Mean Squared Error (MSE):** Penalizes larger prediction errors more heavily.
- **Root Mean Squared Error (RMSE):** Measures prediction error on the same numerical scale as AQI.
- **R-squared (R²):** Measures how well the model explains variation in the target values.

All models will be evaluated using the same held-out test data to make the comparison fair.

## Streamlit Dashboard

The planned dashboard will provide a simple interface for exploring air quality predictions.

Its intended features include:

- Input fields for pollutant concentrations.
- A button to generate an AQI prediction.
- Display of the predicted AQI.
- AQI category display using the appropriate category thresholds.
- Visualizations of historical AQI trends.
- Comparison of model performance.
- Charts showing pollutant distributions and model feature importance.

The dashboard will use the trained model and its associated preprocessing pipeline. Features that are not present in the selected dataset will not be presented as real model inputs unless they are added through a validated extension.

## Technologies Used

- **Python** – Main programming language
- **Google Colab** – Coding, data analysis, and model experimentation
- **Pandas and NumPy** – Data processing
- **Matplotlib** – Data visualization
- **Scikit-learn** – Data preprocessing, regression models, and evaluation
- **XGBoost** – Gradient-boosting model
- **Streamlit** – Interactive web dashboard
- **GitHub** – Version control and project collaboration

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/aqi-prediction-ml.git
cd aqi-prediction-ml
```

Replace `YOUR_USERNAME` with the GitHub username or organization that owns the repository.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Download the dataset from Kaggle and place `city_day.csv` in the location specified in the data preparation notebook.

### 4. Run the notebooks

Open the notebooks in Google Colab or Jupyter Notebook and run the data preparation and model training steps.

### 5. Run the dashboard

Once `app.py` and the trained model artifact are ready, run:

```bash
streamlit run app.py
```

The dashboard will open in your browser. The application requires the trained model and preprocessing pipeline to be available at the configured path.

## Limitations

- The model's predictions depend on the quality and coverage of the historical dataset.
- Missing observations and differences between cities may affect model performance.
- The initial dataset does not contain the weather variables proposed in the original project idea.
- A prediction based on same-day pollutant measurements estimates AQI; it does not necessarily forecast future air quality.
- Model predictions are estimates and should not be treated as official air quality measurements or health advice.

## Expected Outcome

The expected outcome is a working machine learning pipeline that compares different regression algorithms for AQI estimation, along with an interactive dashboard that presents predictions and relevant visualizations.

The project will also provide an opportunity to understand the complete machine learning workflow, from data preparation and model evaluation to application development and deployment.

## Acknowledgements

We acknowledge the original dataset contributors and the Kaggle platform for making the air quality data available for educational and research use.

# Solar Power Time-Series Forecasting

This project forecasts solar power generation using historical power output and weather data.  
It combines machine learning, time-series feature engineering, and Streamlit for interactive visualization and prediction.

---

##  Project Structure

```
.
app/             # Streamlit app
data/            # Cleaned & feature-engineered data
models/          # Trained model (.pkl)
notebooks/       # EDA, modeling, evaluation notebooks
reports/         # Model predictions, plots, residuals
requirements.txt
environment.yml
.gitignore
README.md
```

---

## Features

- Time-based features: hour, month, day of week, weekend flag
- Lag features: AC power lagged by 1-3 hours
- Rolling mean/std: 3h, 6h, 12h smoothing for trends and volatility
- Machine Learning Models: Linear Regression, Random Forest
- Model evaluation using RMSE & MAE
- Streamlit UI for real-time forecasting

---

## Model Performance

| Model             | RMSE   | MAE   |
|------------------|--------|-------|
| Linear Regression | 41.79  | 12.08 |
| Random Forest     | 43.62  | 11.67 |

---

## Key Insights

- `AC_POWER_roll_mean_3h` was the most important feature
- Recent power trends are more predictive than raw weather inputs
- Time based patterns (ie hour of day) also play a significant role

---

## Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Make sure `models/rf_model.pkl` and `reports/model_predictions.csv` exist before launching.

---

## Setup Environment (Conda)

```bash
conda env create -f environment.yml
conda activate solar-forecast-env
```

---

## Contributions

Contributions, suggestions, or new model experiments are welcome.  
Feel free to open an issue or a pull request!

---

## License

 License © 2025 Aravind Raju

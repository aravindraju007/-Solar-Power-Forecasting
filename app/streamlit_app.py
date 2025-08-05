import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

st.set_page_config(page_title="Solar Forecast Dashboard", layout="wide")
st.title("☀️ Solar Power Forecast Dashboard")

# -----------------------------
# Load predictions (historical)
# -----------------------------
@st.cache_data
def load_predictions():
    df = pd.read_csv('../reports/model_predictions.csv', parse_dates=['DATE_TIME'])
    return df.set_index('DATE_TIME')

df = load_predictions()

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load('../models/rf_model.pkl')  # Make sure this file exists

model = load_model()

# -----------------------------
# Forecast Input Section
# -----------------------------
st.markdown("## 🔮 Predict Future AC Power Output")

col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input("Date", datetime.now().date())
    selected_hour = st.slider("Hour of Day", 0, 23)
    ambient_temp = st.number_input("Ambient Temperature (°C)", value=25.0)
    mod_temp = st.number_input("Module Temperature (°C)", value=30.0)

with col2:
    irradiance = st.number_input("Solar Irradiation (W/m²)", value=500.0)
    ac_power_lag1 = st.number_input("Previous Hour AC Power (kW)", value=2.5)
    ac_power_roll = st.number_input("3-Hour Rolling Avg AC Power (kW)", value=2.0)
    plant_id = st.number_input("Plant ID (numeric)", value=4135001)

# Derived features
day = selected_date.day
month = selected_date.month
dayofweek = selected_date.weekday()
is_weekend = 1 if dayofweek >= 5 else 0

# Create input feature vector
input_df = pd.DataFrame({
    'hour': [selected_hour],
    'day': [day],
    'month': [month],
    'dayofweek': [dayofweek],
    'is_weekend': [is_weekend],
    'AMBIENT_TEMPERATURE': [ambient_temp],
    'MODULE_TEMPERATURE': [mod_temp],
    'IRRADIATION': [irradiance],
    'AC_POWER_lag1': [ac_power_lag1],
    'AC_POWER_roll_mean_3h': [ac_power_roll],
    'PLANT_ID': [plant_id]
})

# -----------------------------
# Make prediction
# -----------------------------
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction = round(prediction, 2)
    st.success(f"🔆 Predicted AC Power Output: {prediction} kW")

# -----------------------------
# Visualize Historical Trends
# -----------------------------
st.markdown("## 📈 Historical Predictions Overview")

start_date = st.date_input("Start Date", df.index.min().date())
end_date = st.date_input("End Date", df.index.max().date())

filtered_df = df.loc[str(start_date):str(end_date)]

st.metric("Random Forest - MAE", round(abs((filtered_df['True'] - filtered_df['RF_Pred']).mean()), 2))
st.metric("Random Forest - RMSE", round(((filtered_df['True'] - filtered_df['RF_Pred'])**2).mean()**0.5, 2))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(filtered_df.index, filtered_df['True'], label='True', alpha=0.7)
ax.plot(filtered_df.index, filtered_df['RF_Pred'], label='RF Prediction', alpha=0.7)
ax.set_ylabel("AC Power (kW)")
ax.set_xlabel("Time")
ax.set_title("Historical Forecast")
ax.legend()
st.pyplot(fig)

if st.checkbox("Show raw data"):
    st.write(filtered_df)

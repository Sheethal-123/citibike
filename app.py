import streamlit as st
import pandas as pd
import hopsworks
import sys, os
from datetime import datetime
import matplotlib.pyplot as plt

# 🔧 Add src to path for config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import config

st.set_page_config(layout="wide")
st.title("🚲 Citi Bike Prediction App")
st.markdown("Visualizing predicted ride demand for the top 3 stations in NYC.")

@st.cache_data(ttl=3600)
def fetch_all_predictions():
    project = hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(
        name=config.FEATURE_GROUP_MODEL_PREDICTION,
        version=config.FEATURE_GROUP_MODEL_PREDICTION_VERSION
    )
    df = fg.read()
    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])
    df["year"] = df["pickup_hour"].dt.year
    df["month"] = df["pickup_hour"].dt.month
    df["hour"] = df["pickup_hour"].dt.hour
    df["pickup_location_id"] = df["pickup_location_id"].astype(str)
    return df

try:
    df = fetch_all_predictions()

    # Sidebar filters
    st.sidebar.header("Select a dataset")
    available_years = sorted(df["year"].unique(), reverse=True)
    selected_year = st.sidebar.selectbox("Year", available_years)

    available_months = sorted(df[df["year"] == selected_year]["month"].unique())
    selected_month = st.sidebar.selectbox("Month", available_months)

    # Filtered data
    filtered_df = df[(df["year"] == selected_year) & (df["month"] == selected_month)]
    latest_time = filtered_df["pickup_hour"].max()
    latest_df = filtered_df[filtered_df["pickup_hour"] == latest_time]

    if not latest_df.empty:
        st.success("✅ Data loaded successfully!")
        st.subheader("📄 Sample Data")
        st.dataframe(latest_df)

        st.subheader("📊 Predicted Rides by Location")
        chart_data = latest_df.set_index("pickup_location_id")["predicted_rides"]
        st.bar_chart(chart_data, use_container_width=True)

        st.subheader("🕐 Trips by Hour of Day")
        hourly_grouped = filtered_df.groupby("hour")["predicted_rides"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(hourly_grouped["hour"], hourly_grouped["predicted_rides"], marker="o")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Total Predicted Rides")
        ax.set_title("Predicted Rides Grouped by Hour")
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.warning("⚠️ No data available for the selected year and month.")

except Exception as e:
    st.error(f"❌ Couldn't fetch predictions: {e}")

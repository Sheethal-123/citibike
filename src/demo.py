import streamlit as st
import pandas as pd
import zipfile
import io
import requests
import plotly.express as px
from datetime import datetime

# App configuration
st.set_page_config(layout="wide", page_title="🚴 Citi Bike Dashboard")
st.title("🚴‍♂️ NYC Citi Bike Analytics Dashboard")
st.markdown("""
This interactive dashboard allows you to explore trip data from the Citi Bike system in **Jersey City**.
Select a year and month to load real-world bike ride data and analyze rider patterns across time.
""")

# Sidebar - Dataset controls
st.sidebar.header("📂 Select Dataset")
year = st.sidebar.selectbox("Year", [2023], index=0)
month = st.sidebar.selectbox("Month", list(range(1, 13)), index=datetime.now().month - 2)

# Build URL to fetch data
url = f"https://s3.amazonaws.com/tripdata/JC-{year}{month:02}-citibike-tripdata.csv.zip"
st.sidebar.markdown(f"[🔗 Download raw ZIP]({url})")
st.markdown(f"### 🗂️ Source File: [`JC-{year}{month:02}`]({url})")

@st.cache_data(show_spinner=True)
def load_data(zip_url):
    r = requests.get(zip_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    csv_filename = z.namelist()[0]
    with z.open(csv_filename) as f:
        df = pd.read_csv(f)
    return df

# Load and process
try:
    df = load_data(url)
    st.success("✅ Data loaded successfully!")

    # --- Preprocessing ---
    datetime_col = "started_at" if "started_at" in df.columns else "starttime"
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.dropna(subset=[datetime_col])
    df["hour"] = df[datetime_col].dt.hour
    df["day"] = df[datetime_col].dt.day_name()
    df["date"] = df[datetime_col].dt.date

    # --- Summary Stats ---
    st.subheader("📋 Overview Metrics")
    col1, _, _ = st.columns(3)
    col1.metric("Total Trips", f"{len(df):,}")

    st.markdown("---")

    # --- Trip Volume by Hour ---
    st.subheader("⏰ Hourly Trip Volume")
    hourly_counts = df.groupby("hour").size().reset_index(name="Trips")
    fig_hourly = px.bar(hourly_counts, x="hour", y="Trips", color="Trips", color_continuous_scale="Plasma",
                        title="Trip Frequency by Hour of Day")
    fig_hourly.update_layout(xaxis_title="Hour of Day", yaxis_title="Trip Count")
    st.plotly_chart(fig_hourly, use_container_width=True)

    # --- Daily Trend ---
    st.subheader("📈 Daily Trip Trend")
    daily_counts = df.groupby("date").size().reset_index(name="Trips")
    fig_daily = px.area(daily_counts, x="date", y="Trips", title="Trips Over Days",
                        labels={"date": "Date", "Trips": "Trip Count"},
                        color_discrete_sequence=["#636EFA"])
    st.plotly_chart(fig_daily, use_container_width=True)

    # --- Weekday Heatmap ---
    st.subheader("📊 Weekday vs Hour Heatmap")
    pivot = df.pivot_table(index="hour", columns="day", aggfunc="size", fill_value=0)
    st.dataframe(pivot.style.background_gradient(cmap="YlGnBu", axis=1))

    # --- Raw data viewer ---
    with st.expander("🔍 View Raw Trip Data"):
        st.dataframe(df.head(100))

except Exception as e:
    st.error(f"❌ Failed to load or process data: {e}")
    st.stop()
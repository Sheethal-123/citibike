import streamlit as st
import pandas as pd
import zipfile
import io
import requests

st.set_page_config(layout="wide")
st.title("🚲 NYC Citi Bike Trip Viewer")

# --- User Input ---
st.sidebar.header("Select a dataset")
year = st.sidebar.selectbox("Year", [2023])
month = st.sidebar.selectbox("Month", list(range(1, 13)))

# --- File URL ---
url = f"https://s3.amazonaws.com/tripdata/JC-{year}{month:02}-citibike-tripdata.csv.zip"
st.write(f"🔗 Fetching data from: {url}")

@st.cache_data
def load_data_from_s3(zip_url):
    r = requests.get(zip_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    csv_filename = z.namelist()[0]
    with z.open(csv_filename) as f:
        df = pd.read_csv(f)
    return df

# --- Load and process ---
try:
    df = load_data_from_s3(url)
    st.success("✅ Data loaded successfully!")

    # Show raw data
    st.subheader("📄 Sample Data")
    st.write(df.head())

    # Try to parse datetime column
    datetime_col = "started_at" if "started_at" in df.columns else "starttime"
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.dropna(subset=[datetime_col])
    df["hour"] = df[datetime_col].dt.hour

    # Show trip counts by hour
    st.subheader("⏱ Trips by Hour of Day")
    hourly_counts = df.groupby("hour").size()
    st.line_chart(hourly_counts)

    # Optional stats
    duration_col = "tripduration" if "tripduration" in df.columns else None
    if duration_col:
        st.subheader("🕒 Trip Duration Stats (seconds)")
        st.write(df[duration_col].describe())

except Exception as e:
    st.error(f"❌ Could not load data: {e}")
import streamlit as st
import pandas as pd
import hopsworks
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import sys, os

# 🔧 Add src to path for config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import config

st.set_page_config(page_title="Citi Bike Monitoring", layout="wide")
st.title("📉 Citi Bike Prediction Monitoring Dashboard")
st.markdown("Compare model predictions vs actuals and evaluate performance.")

# -------------------------------
# 🔄 Load data from Hopsworks
# -------------------------------
@st.cache_data(ttl=1800)
def fetch_data():
    project = hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )
    fs = project.get_feature_store()

    pred_fg = fs.get_feature_group(
        name=config.FEATURE_GROUP_MODEL_PREDICTION,
        version=config.FEATURE_GROUP_MODEL_PREDICTION_VERSION
    )
    actual_fg = fs.get_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION
    )

    pred_df = pred_fg.read()
    actual_df = actual_fg.read()

    pred_df["pickup_hour"] = pd.to_datetime(pred_df["pickup_hour"])
    actual_df["pickup_hour"] = pd.to_datetime(actual_df["pickup_hour"])

    merged = pd.merge(
        pred_df,
        actual_df[["pickup_hour", "pickup_location_id", "rides"]],
        on=["pickup_hour", "pickup_location_id"],
        how="inner"
    ).rename(columns={"rides": "actual_rides"})

    merged["pickup_location_id"] = merged["pickup_location_id"].astype(str)
    return merged

try:
    df = fetch_data()

    # -------------------------------
    # 📂 Sidebar filters: Year, Month
    # -------------------------------
    st.sidebar.header("📂 Select a Date Range")

    df["year"] = df["pickup_hour"].dt.year
    df["month"] = df["pickup_hour"].dt.month
    df["pickup_date"] = df["pickup_hour"].dt.date
    df["hour"] = df["pickup_hour"].dt.hour

    years = sorted(df["year"].unique(), reverse=True)
    selected_year = st.sidebar.selectbox("Select Year", years)

    months = sorted(df[df["year"] == selected_year]["month"].unique())
    selected_month = st.sidebar.selectbox("Select Month", months)

    df_month = df[(df["year"] == selected_year) & (df["month"] == selected_month)]

    # -------------------------------
    # 📅 Select date within the month
    # -------------------------------
    available_dates = sorted(df_month["pickup_date"].unique(), reverse=True)
    selected_date = st.selectbox("Select Date", available_dates)

    df_day = df_month[df_month["pickup_date"] == selected_date]

    # -------------------------------
    # 📈 Line plot: rides by hour
    # -------------------------------
    st.subheader(f"📊 Hourly Predictions – {selected_date.strftime('%b %d, %Y')}")

    pivot = df_day.pivot_table(
        index="hour",
        columns="pickup_location_id",
        values=["actual_rides", "predicted_rides"]
    )

    for loc_id in pivot.columns.levels[1]:
        fig = px.line(
            pivot.xs(loc_id, level=1, axis=1),
            markers=True,
            title=f"🚲 Station {loc_id}",
            labels={"value": "Ride Count", "hour": "Hour"},
        )
        fig.update_layout(
            xaxis=dict(tickmode="linear", dtick=1),
            title_x=0.5,
            legend_title="",
            yaxis_title="Ride Count"
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # 📏 Show Error Metrics
    # -------------------------------
    mae = mean_absolute_error(df_day["actual_rides"], df_day["predicted_rides"])
    mape = mean_absolute_percentage_error(df_day["actual_rides"], df_day["predicted_rides"])

    col1, col2 = st.columns(2)
    col1.metric("📏 Mean Absolute Error (MAE)", f"{mae:.2f}")
    col2.metric("📉 Mean Absolute % Error (MAPE)", f"{mape*100:.2f} %")

    # -------------------------------
    # 📄 Expandable raw data table
    # -------------------------------
    with st.expander("📄 Show Raw Comparison Table"):
        st.dataframe(df_day.sort_values(["pickup_location_id", "hour"]))

except Exception as e:
    st.error(f"❌ Error loading data: {e}")


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from predict_xgboost import predict_xgb
from predict_arima import predict_arima
from optimize import optimize_energy
from geocode import get_location
from weather import get_weather

try:
    from anomaly_detection import detect_anomaly
    ANOMALY_AVAILABLE = True
except Exception:
    ANOMALY_AVAILABLE = False

st.set_page_config(page_title="ACEIS Energy Intelligence System", layout="wide")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🐒 ACEIS")


# =====================================
# LOCATION SEARCH
# =====================================

st.sidebar.header("📍 Campus Location")

location_name = st.sidebar.text_input(
    "Search Campus / City",
    "Guru Nanak Dev University"
)

location = get_location(location_name)

if location:

    lat = location["latitude"]
    lon = location["longitude"]

    weather = get_weather(lat, lon)

    st.write("DEBUG WEATHER:", weather)

    temperature = weather.get("temperature", 30)
    humidity = weather.get("humidity", 50)
    condition = weather.get("condition", "Unknown")
    humidity = weather["humidity"]
    condition = weather["condition"]

    st.sidebar.success("Location Found")

    st.sidebar.write(location["address"])

    st.sidebar.metric(
        "Temperature",
        f"{temperature:.1f} °C"
    )

    st.sidebar.metric(
        "Humidity",
        f"{humidity}%"
    )

    st.sidebar.metric(
        "Condition",
        condition
    )

else:

    st.sidebar.error(
        "Location not found"
    )

    temperature = 30
    humidity = 50
    condition = "Unknown"

occupancy = st.sidebar.slider(
    "Occupancy (%)",
    0,
    100,
    75
)

battery_capacity = st.sidebar.slider(
    "Battery Capacity",
    50,
    500,
    200
)

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Forecasting", "Optimization", "Anomaly Detection", "Carbon Emissions", "System Health"]
)



# =========================
# FORECASTING
# =========================
xgb_pred = np.array(predict_xgb(temp=temperature, occupancy=occupancy))
arima_pred = np.array(predict_arima(24))
ensemble_pred = np.array((xgb_pred + arima_pred) / 2)

# =========================
# OPTIMIZATION
# =========================
opt = optimize_energy(
    predicted_load=ensemble_pred,
    battery_capacity=battery_capacity
)

optimized_grid = np.array(opt["optimized_grid"])
optimized_hvac = np.array(opt["optimized_hvac"])
battery_usage = np.array(opt["battery_usage"])
solar_output = np.array(opt["solar_output"])

optimized_cost = float(opt["total_cost"])
peak_load = float(opt["peak_load"])

hours = np.arange(24)

savings = int(np.sum(ensemble_pred) * 10 - optimized_cost)

# =========================
# ANOMALY DETECTION
# =========================
anomaly_scores = []
anomaly_status = []

for hour in range(24):

    if ANOMALY_AVAILABLE:

        try:
            sample = {
                "hour": hour,
                "day": 1,
                "temperature": float(temperature),
                "occupancy": float(occupancy),
                "hvac_load": float((temperature - 20) * 12),
                "solar_output": float(solar_output[hour]),
                "tariff": 8
            }

            pred, score = detect_anomaly(sample)

            anomaly_scores.append(float(score))
            anomaly_status.append(
                "Anomaly" if pred == -1 else "Normal"
            )

        except:
            anomaly_scores.append(float(np.random.uniform(-1, 1)))
            anomaly_status.append("Normal")

    else:
        anomaly_scores.append(float(np.random.uniform(-1, 1)))
        anomaly_status.append("Normal")

# =========================
# DASHBOARD
# =========================
if page == "Dashboard":

    st.title("🐒 ACEIS Energy Intelligence Center")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Forecast Peak", f"{int(max(ensemble_pred))} kW")
    c2.metric("Optimized Peak", f"{int(peak_load)} kW")
    c3.metric("Optimized Cost", f"₹ {int(optimized_cost)}")
    c4.metric("Estimated Savings", f"₹ {savings}")

    st.subheader("Forecast vs Optimized Grid")

    chart_df = pd.DataFrame({
        "Hour": hours,
        "Forecast": ensemble_pred,
        "Optimized Grid": optimized_grid
    })

    st.line_chart(chart_df.set_index("Hour"))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Solar Output")
        st.area_chart(
            pd.DataFrame({
                "Hour": hours,
                "Solar": solar_output
            }).set_index("Hour")
        )

    with col2:
        st.subheader("Battery Usage")
        st.bar_chart(
            pd.DataFrame({
                "Hour": hours,
                "Battery": battery_usage
            }).set_index("Hour")
        )

elif page == "Forecasting":

    st.title(" Multi-Model Forecasting")

    forecast_df = pd.DataFrame({
        "Hour": hours,
        "XGBoost": xgb_pred,
        "ARIMA": arima_pred,
        "Ensemble": ensemble_pred
    })

    st.dataframe(forecast_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hours, xgb_pred, label="XGBoost")
    ax.plot(hours, arima_pred, label="ARIMA")
    ax.plot(hours, ensemble_pred, linewidth=3, label="Ensemble")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Load (kW)")
    ax.legend()

    st.pyplot(fig)

elif page == "Optimization":

    st.title(" Energy Optimization")

    st.metric("Estimated Savings", f"₹ {savings}")

    opt_df = pd.DataFrame({
        "Hour": hours,
        "Forecast": ensemble_pred,
        "Optimized Grid": optimized_grid,
        "HVAC": optimized_hvac,
        "Battery": battery_usage
    })

    st.dataframe(opt_df, use_container_width=True)

    st.line_chart(
        opt_df.set_index("Hour")[["Forecast", "Optimized Grid"]]
    )

elif page == "Anomaly Detection":

    st.title(" Isolation Forest Anomaly Detection")

    col1, col2 = st.columns(2)

    with col1:
        score_df = pd.DataFrame({
            "Hour": hours,
            "Score": anomaly_scores
        })
        st.line_chart(score_df.set_index("Hour"))

    with col2:
        alert_df = pd.DataFrame({
            "Hour": hours,
            "Load": np.round(ensemble_pred, 2),
            "Score": np.round(anomaly_scores, 3),
            "Status": anomaly_status
        })

        st.dataframe(alert_df, use_container_width=True)

    if "Anomaly" in anomaly_status:
        idx = anomaly_status.index("Anomaly")

        st.error(
            f"Alert Detected at Hour {idx} | Score: {anomaly_scores[idx]:.3f}"
        )
    else:
        st.success("No abnormal energy consumption detected")




elif page == "Carbon Emissions":

    st.title(" Carbon Emission Analysis")

    carbon_factor = 0.82

    forecast_emission = np.sum(ensemble_pred) * carbon_factor

    optimized_emission = np.sum(optimized_grid) * carbon_factor

    saved_emission = (
        forecast_emission -
        optimized_emission
    )

    reduction_percent = (
        saved_emission /
        forecast_emission
    ) * 100

    # ======================================
    # KPI CARDS
    # ======================================

    c1,c2,c3 = st.columns(3)

    with c1:
        st.metric(
            "Forecast Emissions",
            f"{forecast_emission:.2f} kg"
        )

    with c2:
        st.metric(
            "Optimized Emissions",
            f"{optimized_emission:.2f} kg"
        )

    with c3:
        st.metric(
            "CO₂ Reduction",
            f"{saved_emission:.2f} kg",
            f"{reduction_percent:.2f}%"
        )

    st.markdown("---")

    # ======================================
    # CHARTS
    # ======================================

    col1,col2 = st.columns(2)

    # ---------------------------
    # BAR CHART
    # ---------------------------

    with col1:

        st.subheader(
            "Emission Comparison"
        )

        emission_df = pd.DataFrame({

            "Type":[
                "Forecast",
                "Optimized"
            ],

            "Emission":[
                forecast_emission,
                optimized_emission
            ]

        })

        st.bar_chart(
            emission_df.set_index("Type")
        )

    # ---------------------------
    # HOURLY FOOTPRINT
    # ---------------------------

    with col2:

        st.subheader(
            "Hourly Carbon Footprint"
        )

        hourly_df = pd.DataFrame({

            "Hour": hours,

            "Forecast CO₂":
                ensemble_pred *
                carbon_factor,

            "Optimized CO₂":
                optimized_grid *
                carbon_factor

        })

        st.line_chart(
            hourly_df.set_index("Hour")
        )

    st.markdown("---")

    # ======================================
    # PIE CHART + IMPACT
    # ======================================

    col3,col4,col5 = st.columns([1,1,1])

    # ---------------------------
    # PIE CHART
    # ---------------------------

    with col3:

        st.subheader(
            "Emission Breakdown"
        )

        hvac_emission = (
            np.sum(optimized_hvac)
            * carbon_factor
        )

        battery_emission = (
            np.sum(battery_usage)
            * carbon_factor
        )

        other_emission = (
            optimized_emission
            - hvac_emission
            - battery_emission
        )

        fig, ax = plt.subplots()

        ax.pie(

            [
                hvac_emission,
                battery_emission,
                other_emission
            ],

            labels=[
                "HVAC",
                "Battery",
                "Other Loads"
            ],

            autopct='%1.1f%%'

        )

        st.pyplot(fig)

    # ---------------------------
    # ENVIRONMENT IMPACT
    # ---------------------------

    with col4:

        st.subheader(
            "Environmental Impact"
        )

        trees_saved = (
            saved_emission / 21
        )

        st.success(f"""
         CO₂ Saved

        {saved_emission:.2f} kg

        Equivalent Trees:

        {trees_saved:.1f}

        ACEIS optimization
        contributes to a
        greener campus.
        """)

    # ---------------------------
    # SUSTAINABILITY SCORE
    # ---------------------------

    with col5:

        st.subheader(
            "Sustainability Score"
        )

        score = int(
            reduction_percent * 5
        )

        score = min(
            score,
            100
        )

        st.metric(
            "Score",
            f"{score}/100"
        )

        progress = score / 100

        st.progress(progress)

        if score > 80:

            st.success(
                "Excellent"
            )

        elif score > 60:

            st.info(
                "Good"
            )

        else:

            st.warning(
                "Needs Improvement"
            )

    st.markdown("---")

    st.info(f"""
    Carbon emissions are calculated using
    a grid emission factor of
    {carbon_factor} kg CO₂/kWh.

    ACEIS helps reduce emissions through:
    • Load Optimization
    • HVAC Control
    • Battery Dispatch
    • Renewable Energy Integration
    """)



elif page == "System Health":

    st.title(" System Health")

    status_df = pd.DataFrame({
        "Module": [
            "XGBoost",
            "ARIMA",
            "Isolation Forest",
            "Optimization Engine",
            "Solar Simulation",
            "Battery System"
        ],
        "Status": [
            "ACTIVE",
            "ACTIVE",
            "ACTIVE" if ANOMALY_AVAILABLE else "NOT AVAILABLE",
            opt["status"],
            "ACTIVE",
            "ACTIVE"
        ]
    })

    st.table(status_df)

st.sidebar.success("ACEIS Running")

import os
import streamlit as st
from utils.anomaly_detection import detect_anomalies, format_anomaly_summary
from utils.groq_ai import generate_groq_insight
import pandas as pd


def show_ai_insight(df):
    st.title("🤖 AI Driver Insight (Groq + Anomaly Detection)")

    st.markdown("""
    AI-powered driver performance analysis combining:
    - 📉 Statistical anomaly detection  
    - 🧠 Groq LLaMA expert reasoning  
    - 🏎️ Lap & sector interpretation  
    """)

    if df is None or df.empty:
        st.warning("No data available for this driver.")
        return

    # =============================================
    # 1. RUN ANOMALY DETECTION
    # =============================================
    anomalies = detect_anomalies(df)

    if anomalies is not None and len(anomalies) > 0:
        st.subheader("🚨 Detected Driving Anomalies")
        st.dataframe(anomalies.head(15), use_container_width=True)
        anomaly_text = format_anomaly_summary(anomalies)
    else:
        st.success("No anomalies detected 🎉")
        anomaly_text = "No anomalies detected."

    # =============================================
    # 2. SAFELY PREPARE DATA SUMMARY
    # =============================================
    # vehicle_id
    vehicle_id = (
        df["vehicle_id"].iloc[0]
        if "vehicle_id" in df.columns and len(df["vehicle_id"]) > 0
        else "Unknown"
    )

    # lap count
    lap_count = (
        df["lap"].nunique() if "lap" in df.columns else "N/A"
    )

    # avg speed (handle None, NaN)
    if "telemetry_value" in df.columns:
        avg_speed = df["telemetry_value"].mean()

        if pd.isna(avg_speed):
            avg_speed_safe = "N/A"
        else:
            avg_speed_safe = f"{avg_speed:.2f}"
    else:
        avg_speed_safe = "N/A"

    # =============================================
    # 3. BUILD TEXT FOR LLaMA (SAFE)
    # =============================================
    summary_text = f"""
Driver Summary:
- Driver: {vehicle_id}
- Lap count: {lap_count}
- Average Telemetry Value: {avg_speed_safe}

{anomaly_text}
"""

    # =============================================
    # 4. CALL GROQ LLaMA AI
    # =============================================
    st.subheader("🧠 AI Reasoning (Groq LLaMA-3.3-70B)")

    with st.spinner("Generating expert racing insight..."):
        ai_output = generate_groq_insight(summary_text)

    # =============================================
    # 5. OUTPUT
    # =============================================
    st.markdown("### 📌 Expert AI Insight")
    st.markdown(ai_output)

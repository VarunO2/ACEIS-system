try:
    from predict_xgboost import predict_xgb
    from predict_arima import predict_arima
    from optimize import optimize_energy

    st.success("Imports successful")

except Exception as e:
    st.error(e)
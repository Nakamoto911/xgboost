import streamlit as st
import os

st.set_page_config(
    page_title="JM-XGBoost Backtester",
    page_icon="🚀",
    layout="wide",
)

st.title("Generative AI & Quant Backtesting Control Center")

st.markdown("""
Welcome to the JM-XGBoost strategy platform. This tool allows for robust testing, diagnostics, and LLM-assisted analysis of your trading pipeline.

### Use the Sidebar to Navigate:

1. **🚀 Performance Tracker**: A highly optimized, fast, and parameter-heavy dashboard. Use this for rapid iteration and testing of strategy variations without the overhead of heavy ML diagnostics.
2. **📊 Model Analysis**: A comprehensive dashboard featuring full model analysis including Feature Importances, SHAP values, regime classification details, and tree visualizations.
3. **🛠️ Diagnostics Launcher**: Control center to run background scripts like `run_experiments.py` or generate reports for LLM consumption.
""")

st.info("Select a module from the sidebar to begin.")

import streamlit as st

st.set_page_config(layout="wide", page_title="Portfolio Construction")

pg = st.navigation([
    st.Page("portfolio_construction.py", title="Portfolio Construction", icon="📈", default=True),
    st.Page("pages/1_📊_Model_Analysis.py",          title="Model Analysis",        icon="📊"),
    st.Page("pages/2_🛠️_Diagnostics_Launcher.py",   title="Diagnostics Launcher",  icon="🛠️"),
    st.Page("pages/3_🔍_Data_Quality_Audit.py",      title="Data Quality Audit",    icon="🔍"),
])
pg.run()

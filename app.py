import streamlit as st

st.set_page_config(
    page_title="Wave Packet Simulation",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation([
    st.Page("main_app.py", title="Wave Packet Simulation", icon="🌊", default=True),
    st.Page("pages/1_Theory_and_Working.py", title="Theory and Working", icon="📘"),
    st.Page("pages/2_Superpose_Waves.py", title="Superpose Waves", icon="〰️"),
])
pg.run()

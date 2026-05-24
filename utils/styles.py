import streamlit as st


def inject_css():
    st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 4rem;
    padding-bottom: 2rem;
}

.big-title {
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    color: #B0B0B0;
    margin-bottom: 20px;
}

.prediction-card {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
    padding: 30px;
    border-radius: 20px;
    color: black;
    text-align: center;
    margin-top: 20px;
}

.metric-card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

.section-card {
    background-color: #1A1C24;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

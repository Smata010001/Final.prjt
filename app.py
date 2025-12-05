import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Pro‑Sole: Student Performance Analytics",
    page_icon="📚",
)

st.title("Pro‑Sole: Student Performance Analytics")
st.markdown("""
Welcome! This app helps schools understand how students’ backgrounds and support systems
relate to their exam performance in **math, reading, and writing**.

- **Explore the data** on the *Data Description* and *Data Visualization* pages.  
- **Predict performance** for a given student on the *Model Prediction* page.  
- Use the navigation in the sidebar to move between pages.
""")

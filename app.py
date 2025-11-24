import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI Impact on Jobs 2030", page_icon="🤖")
st.title("AI_Impact_on_Jobs_2030")
st.markdown("""
Welcome! This app explores the impact of AI on job automation by 2030.
- **Goal:** Predict the probability that different job titles will be automated, using machine learning.
- Use the sidebar to navigate through the analysis, prediction, and explainability tools.
""")

# Load data for display
@st.cache_data
def load_data():
    df = pd.read_csv('AI_Impact_on_Jobs_2030.csv')
    return df

df = load_data()

if st.checkbox("Show raw data table"):
    st.write(df.head(10))

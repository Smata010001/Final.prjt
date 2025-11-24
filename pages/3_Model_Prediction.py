import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
import joblib

st.title("Automation Probability Prediction")

# --- Load Data and Models ---
@st.cache_data
def load_data():
    df = pd.read_csv('AI_Impact_on_Jobs_2030.csv')
    return df

df = load_data()

# Optional: Pre-train your models and save with joblib if you want fast loading
@st.cache_resource
def load_model(model_name):
    # Example: You can use joblib to load saved models, if you've exported after training
    # return joblib.load(f"{model_name}.joblib")
    # For demo, we retrain simple models inline (replace with load for actual project)
    features = [
        'Average_Salary', 'Years_Experience', 'AI_Exposure_Index',
        'Tech_Growth_Factor'
        # add more if needed, but don't include target or non-numeric
    ]
    X = df[features]
    y = df['Automation_Probability_2030']
    if model_name == "Linear Regression":
        model = LinearRegression().fit(X, y)
    else:
        model = Ridge(alpha=1.0).fit(X, y)
    return model, features

# --- Side panel for input ---
st.sidebar.header("Select Model and Job Features")

model_option = st.sidebar.selectbox(
    "Select model", 
    ("Linear Regression", "Ridge Regression")
)

sample = {}
sample['Average_Salary'] = st.sidebar.slider(
    "Average Salary", 
    int(df['Average_Salary'].min()), int(df['Average_Salary'].max()), 
    int(df['Average_Salary'].mean())
)
sample['Years_Experience'] = st.sidebar.slider(
    "Years Experience", 
    int(df['Years_Experience'].min()), int(df['Years_Experience'].max()), 
    int(df['Years_Experience'].mean())
)
sample['AI_Exposure_Index'] = st.sidebar.slider(
    "AI Exposure Index", 
    float(df['AI_Exposure_Index'].min()), float(df['AI_Exposure_Index'].max()), 
    float(df['AI_Exposure_Index'].mean())
)
sample['Tech_Growth_Factor'] = st.sidebar.slider(
    "Tech Growth Factor", 
    float(df['Tech_Growth_Factor'].min()), float(df['Tech_Growth_Factor'].max()), 
    float(df['Tech_Growth_Factor'].mean())
)
# Add more inputs for skills or categories as needed

if st.sidebar.button("Predict"):
    model, feature_list = load_model(model_option)
    input_array = np.array([sample[f] for f in feature_list]).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    st.success(f"Predicted automation probability by 2030: {prediction:.2f} (0 = not likely, 1 = very likely)")

st.markdown(
    """
    **How does this work?**  
    - Select your model and fill in the job or skill factors.
    - Click 'Predict' to see the probability that a job with those characteristics may be automated by 2030.
    - Try adjusting different features to see their impact!
    """
)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

st.title("Automation Probability Predictor")

@st.cache_data
def load_data():
    df = pd.read_csv('AI_Impact_on_Jobs_2030.csv')
    return df

df = load_data()

# Define feature columns used for prediction
features = [
    "Average_Salary", "Years_Experience", "AI_Exposure_Index", "Tech_Growth_Factor"
    # Add Skill columns if you want, e.g. "Skill_1", "Skill_2"
]
target = "Automation_Probability_2030"

# Split data (for training quick demo models—replace with your best feature engineering/model for final version)
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar UI for model selection
model_choice = st.sidebar.selectbox(
    "Choose regression model:",
    ("Linear Regression", "Ridge Regression")
)

# UI sliders for main feature input (adjust min/max/range as needed)
st.sidebar.header("Enter job features for prediction:")
user_input = {}
for col in features:
    col_min = float(df[col].min())
    col_max = float(df[col].max())
    col_mean = float(df[col].mean())
    if np.issubdtype(df[col].dtype, np.integer):
        user_input[col] = st.sidebar.slider(col, int(col_min), int(col_max), int(col_mean))
    else:
        user_input[col] = st.sidebar.slider(col, col_min, col_max, float(col_mean), step=0.01)

input_array = np.array([user_input[f] for f in features]).reshape(1, -1)

if model_choice == "Linear Regression":
    model = LinearRegression()
else:
    model = Ridge(alpha=1.0)

model.fit(X_train, y_train)  # Train model
prediction = model.predict(input_array)[0]

st.header("Predicted Automation Risk")
st.write(f"For a job with these features, our {model_choice} model predicts an automation probability by 2030 of: **{prediction:.2f}**")

# Optionally show predicted vs actual for one of the test jobs (great for demo!)
st.subheader("Sample Prediction vs Actual (from test set)")
idx = np.random.choice(X_test.index)
sample_features = X_test.loc[idx].values.reshape(1, -1)
sample_true = y_test.loc[idx]
sample_pred = model.predict(sample_features)[0]
st.write(f"Sample job features: {X_test.loc[idx].to_dict()}")
st.write(f"Actual automation probability: {sample_true:.2f}")
st.write(f"Predicted (by {model_choice}): {sample_pred:.2f}")

st.info("Try different model types and job features in the sidebar to see their impact on predicted automation probability!")

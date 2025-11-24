import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.title("🤖 Automation Probability Predictor")

@st.cache_data
def load_data():
    return pd.read_csv("AI_Impact_on_Jobs_2030.csv")

df = load_data()

# ----- Features & target -----
features = [
    "Average_Salary",
    "Years_Experience",
    "AI_Exposure_Index",
    "Tech_Growth_Factor",
]
target = "Automation_Probability_2030"

X = df[features]
y = df[target]

@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
    }

    trained = {}
    metrics = {}

    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        trained[name] = m
        metrics[name] = {"mae": mae, "r2": r2}

    return trained, metrics

models, metrics = train_models(X, y)

st.write(
    """
    Use this page to predict the **probability that a job will be automated by 2030** 
    based on different job characteristics.
    """
)

# ----- Model choice -----
model_choice = st.selectbox(
    "Choose a regression model:",
    ("Linear Regression", "Ridge Regression")
)
model = models[model_choice]
metric = metrics[model_choice]

# ----- Model metrics -----
col_mae, col_r2 = st.columns(2)
with col_mae:
    st.metric("Mean Absolute Error (MAE)", f"{metric['mae']:.3f}")
with col_r2:
    st.metric("R² Score", f"{metric['r2']:.3f}")

st.markdown("---")

# ----- Sliders for input -----
st.subheader("Enter job features to predict automation risk")

user_input = {}
c1, c2 = st.columns(2)

for i, col in enumerate(features):
    col_min = float(df[col].min())
    col_max = float(df[col].max())
    col_mean = float(df[col].mean())

    slider_kwargs = {
        "label": col,
        "min_value": col_min,
        "max_value": col_max,
        "value": col_mean,
        "step": 0.01,
    }

    if not np.issubdtype(df[col].dtype, np.floating):
        slider_kwargs["min_value"] = int(col_min)
        slider_kwargs["max_value"] = int(col_max)
        slider_kwargs["value"] = int(col_mean)
        slider_kwargs["step"] = 1

    # Place sliders in two columns
    if i % 2 == 0:
        with c1:
            user_input[col] = st.slider(**slider_kwargs)
    else:
        with c2:
            user_input[col] = st.slider(**slider_kwargs)

input_array = np.array([user_input[f] for f in features]).reshape(1, -1)

st.markdown("---")

# ----- Predict button -----
if st.button("🔮 Predict Automation Probability"):
    prediction = model.predict(input_array)[0]

    st.subheader("Predicted Automation Risk (2030)")
    st.write(
        f"The **{model_choice}** model predicts an automation probability of "
        f"**{prediction:.2f}** for a job with these characteristics."
    )

st.info("Adjust the sliders above and click Predict to explore different scenarios.")

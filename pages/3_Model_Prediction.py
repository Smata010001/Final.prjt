import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

sns.set_theme(style="whitegrid")

st.title("📈 Linear Regression Check – Global AI Content Impact")

@st.cache_data
def load_data():
    return pd.read_csv("Global_AI_Content_Impact_Dataset.csv")

df = load_data()

st.write(
    """
    This page lets us run a **linear regression** on the new dataset to see whether
    there is a meaningful relationship between variables (i.e., whether the data
    behaves like “real” data or more like random noise).
    """
)

st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Select target and features
# -----------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

st.subheader("1️⃣ Choose Target and Features")

target_col = st.selectbox(
    "Select the target variable (what you want to predict):",
    numeric_cols,
    index=numeric_cols.index("Job Loss Due to AI (%)") if "Job Loss Due to AI (%)" in numeric_cols else 0
)

feature_options = [c for c in numeric_cols if c != target_col]

default_features = feature_options  # start with all others

selected_features = st.multiselect(
    "Select feature columns (inputs for the model):",
    feature_options,
    default=default_features
)

if len(selected_features) == 0:
    st.warning("Please select at least one feature to run the regression.")
    st.stop()

X = df[selected_features]
y = df[target_col]

# -----------------------------
# Train/test split + model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("2️⃣ Model Performance")

col1, col2 = st.columns(2)
with col1:
    st.metric("Mean Absolute Error (MAE)", f"{mae:.3f}")
with col2:
    st.metric("R² Score", f"{r2:.3f}")

st.caption(
    "- **MAE** is the average error between predicted and actual values (lower is better).\n"
    "- **R²** close to 1 means strong linear relationship, near 0 means weak/none, and negative means the model is worse than just predicting the mean."
)

# -----------------------------
# Actual vs predicted plot
# -----------------------------
st.subheader("3️⃣ Actual vs Predicted")

fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle="--", color="gray")
ax.set_xlabel("Actual values")
ax.set_ylabel("Predicted values")
ax.set_title(f"Actual vs Predicted – Target: {target_col}")
st.pyplot(fig)

st.caption(
    "If the data has a strong linear pattern, points will lie close to the dashed line."
)

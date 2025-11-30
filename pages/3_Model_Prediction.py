# pages/2_🔮_Model_Prediction.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Student Performance – Prediction",
    layout="wide",
)

@st.cache_data
def load_data():
    df = pd.read_csv("StudentsPerformance.csv")
    df["overall_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)
    return df

@st.cache_resource
def train_models(df):

    # Features and target
    X = df[["gender",
            "race/ethnicity",
            "parental level of education",
            "lunch",
            "test preparation course"]]
    y = df["overall_score"]

    cat_cols = X.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), cat_cols)
        ]
    )

    # Model 1: simple linear regression
    model_lr = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression())
        ]
    )

    # Model 2: Ridge regression (slight regularization)
    model_ridge = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", Ridge(alpha=1.0))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_lr.fit(X_train, y_train)
    model_ridge.fit(X_train, y_train)

    # Evaluate
    preds_lr = model_lr.predict(X_test)
    preds_ridge = model_ridge.predict(X_test)

    metrics = {
        "Linear regression": {
            "MAE": mean_absolute_error(y_test, preds_lr),
            "R2": r2_score(y_test, preds_lr),
        },
        "Ridge regression": {
            "MAE": mean_absolute_error(y_test, preds_ridge),
            "R2": r2_score(y_test, preds_ridge),
        },
    }

    return model_lr, model_ridge, metrics, X_train, y_train

df = load_data()
model_lr, model_ridge, metrics, X_train, y_train = train_models(df)

# ---------- PAGE HEADER ----------
st.title("Predict a Student’s Exam Performance")
st.caption(
    "Create a student profile and estimate expected exam performance using two different models."
)

# ---------- USER INPUT: STUDENT PROFILE ----------
st.subheader("1. Describe the student")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", sorted(df["gender"].unique()))
    race = st.selectbox("Race / Ethnicity", sorted(df["race/ethnicity"].unique()))
with col2:
    parent_ed = st.selectbox(
        "Parental education level",
        sorted(df["parental level of education"].unique()),
    )
    lunch = st.selectbox("Lunch type", sorted(df["lunch"].unique()))
with col3:
    test_prep = st.selectbox(
        "Test preparation course",
        sorted(df["test preparation course"].unique()),
    )

input_dict = {
    "gender": gender,
    "race/ethnicity": race,
    "parental level of education": parent_ed,
    "lunch": lunch,
    "test preparation course": test_prep,
}
input_df = pd.DataFrame([input_dict])

# ---------- MODEL SELECTION ----------
st.subheader("2. Choose a model")

model_name = st.radio(
    "Select prediction model",
    ["Linear regression (baseline)", "Ridge regression (enhanced)"],
    horizontal=True,
)

if model_name.startswith("Linear"):
    current_model = model_lr
    model_key = "Linear regression"
else:
    current_model = model_ridge
    model_key = "Ridge regression"

# ---------- PREDICTION ----------
st.subheader("3. Predicted scores for this student")

if st.button("Predict exam performance"):
    # Predict overall score
    overall_pred = current_model.predict(input_df)[0]

    # To get subject-level predictions in a simple way,
    # assume each subject is close to overall with small adjustments
    # based on training set mean differences.
    subj_means = df[["math score", "reading score", "writing score"]].mean()
    overall_mean = df["overall_score"].mean()
    adjustments = subj_means - overall_mean

    pred_math = overall_pred + adjustments["math score"]
    pred_read = overall_pred + adjustments["reading score"]
    pred_write = overall_pred + adjustments["writing score"]

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Predicted math score", f"{pred_math:.1f}")
    with kpi2:
        st.metric("Predicted reading score", f"{pred_read:.1f}")
    with kpi3:
        st.metric("Predicted writing score", f"{pred_write:.1f}")
    with kpi4:
        st.metric("Predicted overall score", f"{overall_pred:.1f}")

    # Bar chart of predicted scores
    st.markdown("### Predicted subject scores")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.barplot(
        x=["Math", "Reading", "Writing"],
        y=[pred_math, pred_read, pred_write],
        palette="Blues",
        ax=ax,
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Predicted score")
    st.pyplot(fig, use_container_width=True)

# ---------- MODEL PERFORMANCE SUMMARY ----------
st.markdown("---")
st.subheader("4. How well do the models perform overall?")

metrics_df = pd.DataFrame(metrics).T.round(3)
st.dataframe(metrics_df, use_container_width=True)

st.caption(
    "MAE = average absolute error in predicted overall score (lower is better). "
    "R² = proportion of variance explained (closer to 1 is better)."
)


# ---------- WHAT-IF SCENARIOS ----------
st.markdown("---")
st.subheader("5. What-if scenarios for this student")

st.write(
    "Compare how support changes (lunch type, test prep) might affect the predicted overall score "
    "for the same student profile."
)

scenario_base = input_dict.copy()
scenarios = []

# Baseline
scenarios.append({"Scenario": "Current profile", **scenario_base})

# If test prep completed
if scenario_base["test preparation course"] == "none":
    s = scenario_base.copy()
    s["test preparation course"] = "completed"
    scenarios.append({"Scenario": "Completes test prep", **s})
else:
    s = scenario_base.copy()
    s["test preparation course"] = "none"
    scenarios.append({"Scenario": "Does NOT do test prep", **s})

# If lunch changed
if scenario_base["lunch"] == "standard":
    s = scenario_base.copy()
    s["lunch"] = "free/reduced"
    scenarios.append({"Scenario": "Moves to free/reduced lunch", **s})
else:
    s = scenario_base.copy()
    s["lunch"] = "standard"
    scenarios.append({"Scenario": "Moves to standard lunch", **s})

scenarios_df = pd.DataFrame(scenarios)

# Predict for each scenario using the enhanced model
scenario_preds = model_ridge.predict(
    scenarios_df[
        ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
    ]
)
scenarios_df["Predicted overall score"] = scenario_preds

st.dataframe(
    scenarios_df[["Scenario", "lunch", "test preparation course", "Predicted overall score"]]
    .round(1),
    use_container_width=True,
)

fig, ax = plt.subplots(figsize=(6, 3))
sns.barplot(
    data=scenarios_df,
    x="Scenario",
    y="Predicted overall score",
    palette="muted",
    ax=ax,
)
ax.set_ylim(0, 100)
ax.set_ylabel("Predicted overall score")
ax.set_xlabel("")
ax.set_title("Effect of support changes on predicted overall score")
plt.xticks(rotation=15, ha="right")
st.pyplot(fig, use_container_width=True)

# ---------- SIMPLE COEFFICIENT VIEW ----------
st.markdown("---")
st.subheader("6. Which factors matter most in the enhanced model?")

# Extract coefficients from the Ridge model for a quick interpretability view
ohe = model_ridge.named_steps["preprocess"].named_transformers_["cat"]
feature_names = ohe.get_feature_names_out(
    ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
)
coefs = model_ridge.named_steps["model"].coef_

coef_df = pd.DataFrame(
    {"Feature": feature_names, "Coefficient": coefs}
).sort_values(by="Coefficient", key=np.abs, ascending=False)

top_coef = coef_df.head(8)

fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(
    data=top_coef,
    x="Coefficient",
    y="Feature",
    palette="coolwarm",
    ax=ax,
)
ax.set_title("Top factors influencing predicted overall score")
st.pyplot(fig, use_container_width=True)

# pages/2_🔮_Model_Prediction.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score   # CHANGED
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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

    # Model 2: Random Forest with regularization
    # (shallower trees + larger leaves to reduce overfitting)
    model_rf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_depth=6,          # limit depth
                min_samples_leaf=5,   # require more samples per leaf
                max_features="sqrt",  # fewer features per split
                random_state=42,
                n_jobs=-1,
            ))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_lr.fit(X_train, y_train)
    model_rf.fit(X_train, y_train)

    # Evaluate on test set
    preds_lr = model_lr.predict(X_test)
    preds_rf = model_rf.predict(X_test)

    mae_lr = mean_absolute_error(y_test, preds_lr)
    r2_lr = r2_score(y_test, preds_lr)

    mae_rf = mean_absolute_error(y_test, preds_rf)
    r2_rf = r2_score(y_test, preds_rf)

    # Simple 5-fold CV on the TRAINING data for more stable R2
    cv_scores_rf = cross_val_score(
        model_rf, X_train, y_train,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    cv_r2_rf = cv_scores_rf.mean()

    metrics = {
        "Linear regression": {
            "MAE": mae_lr,
            "R2": r2_lr,
        },
        "Random forest": {
            "MAE": mae_rf,
            "R2": r2_rf,
            "CV_R2 (train, 5-fold)": cv_r2_rf,   # extra info
        },
    }

    return model_lr, model_rf, metrics, X_train, y_train

df = load_data()
model_lr, model_rf, metrics, X_train, y_train = train_models(df)

# ---------- PAGE HEADER ----------
st.title("🔮 Student Performance Predictor")
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
    ["Linear regression", "Random forest"],
    horizontal=True,
)

if model_name.startswith("Linear"):
    current_model = model_lr
    model_key = "Linear regression"
else:
    current_model = model_rf
    model_key = "Random forest"

# ---------- PREDICTION ----------
st.subheader("3. Predicted scores for this student")

if st.button("Predict exam performance"):
    overall_pred = current_model.predict(input_df)[0]

    subj_means = df[["math score", "reading score", "writing score"]].mean()
    overall_mean = df["overall_score"].mean()
    adjustments = subj_means - overall_mean

    pred_math = overall_pred + adjustments["math score"]
    pred_read = overall_pred + adjustments["reading score"]
    pred_write = overall_pred + adjustments["writing score"]

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("📐 Math", f"{pred_math:.1f}")
    with kpi2:
        st.metric("📖 Reading", f"{pred_read:.1f}")
    with kpi3:
        st.metric("✍️ Writing", f"{pred_write:.1f}")
    with kpi4:
        st.metric("⭐ Overall", f"{overall_pred:.1f}")

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
    "R² = proportion of variance explained on the test set (closer to 1 is better). "
    "CV_R2 shows 5-fold cross-validated R² on the training data for the random forest."
)

# ---------- WHAT-IF SCENARIOS ----------
st.markdown("---")
st.subheader("💡  What-if scenarios for this student")

st.write(
    "Compare how support changes (lunch type, test prep) might affect the predicted overall score "
    "for the same student profile."
)

scenario_base = input_dict.copy()
scenarios = []

scenarios.append({"Scenario": "Current profile", **scenario_base})

if scenario_base["test preparation course"] == "none":
    s = scenario_base.copy()
    s["test preparation course"] = "completed"
    scenarios.append({"Scenario": "Completes test prep", **s})
else:
    s = scenario_base.copy()
    s["test preparation course"] = "none"
    scenarios.append({"Scenario": "Does NOT do test prep", **s})

if scenario_base["lunch"] == "standard":
    s = scenario_base.copy()
    s["lunch"] = "free/reduced"
    scenarios.append({"Scenario": "Moves to free/reduced lunch", **s})
else:
    s = scenario_base.copy()
    s["lunch"] = "standard"
    scenarios.append({"Scenario": "Moves to standard lunch", **s})

scenarios_df = pd.DataFrame(scenarios)

scenario_preds = model_rf.predict(
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

# ---------- IMPORTANCE VIEW ----------
st.markdown("---")
st.subheader("6. Which factors matter most in the random forest?")
st.info(
    "Higher importance values mean the feature is more useful for splitting the trees in the forest.",
    icon="🧠",
)

ohe = model_rf.named_steps["preprocess"].named_transformers_["cat"]
feature_names = ohe.get_feature_names_out(
    ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
)

importances = model_rf.named_steps["model"].feature_importances_

importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": importances}
).sort_values(by="Importance", ascending=False)

top_importance = importance_df.head(8)

fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(
    data=top_importance,
    x="Importance",
    y="Feature",
    palette="viridis",
    ax=ax,
)
ax.set_title("Most important factors in the random forest")
st.pyplot(fig, use_container_width=True)

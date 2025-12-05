# pages/1_📊_Data_Visualization.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Student Performance – Data Insights",
    layout="wide",
)

@st.cache_data
def load_data():
    df = pd.read_csv("StudentsPerformance.csv")
    df["overall_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)
    return df

df = load_data()

st.image(
    "images/data_vizban.png",
    use_container_width=True,
)
# ---------- PAGE TITLE ----------
st.title("Explore Student Test Results")
st.caption(
    "Understand how student background and school support relate to math, reading, and writing scores."
)

# ---------- KPI METRICS ----------
kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.metric("Avg math score", f"{df['math score'].mean():.1f}")
with kpi_cols[1]:
    st.metric("Avg reading score", f"{df['reading score'].mean():.1f}")
with kpi_cols[2]:
    st.metric("Avg writing score", f"{df['writing score'].mean():.1f}")
with kpi_cols[3]:
    st.metric("Avg overall score", f"{df['overall_score'].mean():.1f}")

st.markdown("---")

# ---------- SCORE DISTRIBUTIONS ----------
st.subheader("Score distributions")

dist_col1, dist_col2, dist_col3 = st.columns(3)
score_cols = ["math score", "reading score", "writing score"]
titles = ["Math score", "Reading score", "Writing score"]

for col, score, title in zip([dist_col1, dist_col2, dist_col3], score_cols, titles):
    with col:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.histplot(
            data=df,
            x=score,
            bins=15,
            kde=True,
            color="#4C72B0",
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Score (0–100)")
        ax.set_ylabel("Count")
        st.pyplot(fig, use_container_width=True)

st.markdown("---")

# ---------- IMPACT OF SUPPORT FACTORS ----------
st.subheader("How support programs relate to scores")

support_tab1, support_tab2 = st.tabs(["Test preparation", "Lunch type"])

with support_tab1:
    fig, ax = plt.subplots(figsize=(6, 4))
    tmp = (
        df.melt(
            id_vars=["test preparation course"],
            value_vars=["math score", "reading score", "writing score"],
            var_name="Subject",
            value_name="Score",
        )
    )
    sns.barplot(
        data=tmp,
        x="Subject",
        y="Score",
        hue="test preparation course",
        estimator=np.mean,
        errorbar="ci",
        palette="Set2",
        ax=ax,
    )
    ax.set_ylabel("Average score")
    ax.set_xlabel("")
    ax.set_title("Average score by test preparation status")
    st.pyplot(fig, use_container_width=True)
    st.caption("Students who complete the test-prep course tend to score higher on average.")

with support_tab2:
    fig, ax = plt.subplots(figsize=(6, 4))
    tmp = (
        df.melt(
            id_vars=["lunch"],
            value_vars=["math score", "reading score", "writing score"],
            var_name="Subject",
            value_name="Score",
        )
    )
    sns.barplot(
        data=tmp,
        x="Subject",
        y="Score",
        hue="lunch",
        estimator=np.mean,
        errorbar="ci",
        palette="Set1",
        ax=ax,
    )
    ax.set_ylabel("Average score")
    ax.set_xlabel("")
    ax.set_title("Average score by lunch type")
    st.pyplot(fig, use_container_width=True)
    st.caption("Lunch type (a proxy for economic support) is linked to differences in scores.")

st.markdown("---")

# ---------- PARENTAL EDUCATION ----------
st.subheader("Overall performance and parental education")

fig, ax = plt.subplots(figsize=(8, 4))
order = (
    df.groupby("parental level of education")["overall_score"]
    .mean()
    .sort_values()
    .index
)
sns.barplot(
    data=df,
    y="parental level of education",
    x="overall_score",
    order=order,
    estimator=np.mean,
    errorbar="ci",
    palette="Blues_r",
    ax=ax,
)
ax.set_xlabel("Average overall score")
ax.set_ylabel("Parental education level")
st.pyplot(fig, use_container_width=True)

st.markdown("---")

# ---------- GENDER & RACE ROLES ----------
st.subheader("Scores by gender and race/ethnicity")

fig, ax = plt.subplots(figsize=(7, 4))
tmp = (
    df.groupby(["race/ethnicity", "gender"])[
        ["math score", "reading score", "writing score"]
    ]
    .mean()
    .reset_index()
)
tmp["overall_score"] = tmp[["math score", "reading score", "writing score"]].mean(axis=1)

sns.barplot(
    data=tmp,
    x="race/ethnicity",
    y="overall_score",
    hue="gender",
    palette="Set2",
    ax=ax,
)
ax.set_xlabel("Race / Ethnicity group")
ax.set_ylabel("Average overall score")
ax.set_title("Average overall score by gender and race/ethnicity")
st.pyplot(fig, use_container_width=True)
st.caption("Shows performance differences between genders within each race/ethnicity group.")

st.markdown("---")

st.subheader("Race/ethnicity and test preparation")

fig, ax = plt.subplots(figsize=(7, 4))
tmp = (
    df.groupby(["race/ethnicity", "test preparation course"])["overall_score"]
    .mean()
    .reset_index()
)
sns.barplot(
    data=tmp,
    x="race/ethnicity",
    y="overall_score",
    hue="test preparation course",
    palette="Paired",
    ax=ax,
)
ax.set_xlabel("Race / Ethnicity group")
ax.set_ylabel("Average overall score")
ax.set_title("Average overall score by race/ethnicity and test prep")
st.pyplot(fig, use_container_width=True)

st.markdown("---")


# ---------- CORRELATION HEATMAP ----------
st.markdown("---")
st.subheader("Correlation between scores")

corr_cols = ["math score", "reading score", "writing score", "overall_score"]
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=0,
    vmax=1,
    square=True,
    cbar_kws={"shrink": 0.8},
    ax=ax,
)
ax.set_title("Score correlation heatmap")
st.pyplot(fig, use_container_width=True)

st.caption("Darker squares mean a stronger relationship between two scores.")

# ---------- BOX PLOT BY RACE ----------
st.markdown("---")
st.subheader("Score spread by race/ethnicity")

fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(
    data=df,
    x="race/ethnicity",
    y="overall_score",
    palette="pastel",
    ax=ax,
)
ax.set_xlabel("Race / Ethnicity group")
ax.set_ylabel("Overall score")
ax.set_title("Distribution of overall scores by race/ethnicity")
st.pyplot(fig, use_container_width=True)


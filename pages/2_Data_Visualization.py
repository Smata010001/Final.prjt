import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

st.title("📊 Data Visualization – AI Impact on Jobs (2030)")

@st.cache_data
def load_data():
    return pd.read_csv("AI_Impact_on_Jobs_2030.csv")

df = load_data()


# ---------------------------------------------------------------------
# Univariate Distributions
# ---------------------------------------------------------------------
st.subheader("1️⃣ Univariate Distributions")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Automation Probability (2030)**")
    fig, ax = plt.subplots()
    sns.histplot(
        data=df,
        x="Automation_Probability_2030",
        hue="Risk_Category",
        kde=True,
        palette="Set2",
        ax=ax
    )
    ax.set_title("Distribution of Automation Probability (2030)")
    st.pyplot(fig)
    st.caption("Shows how risk categories differ in automation probability.")

with col2:
    st.markdown("**AI Exposure Index**")
    fig, ax = plt.subplots()
    sns.histplot(
        data=df,
        x="AI_Exposure_Index",
        hue="Risk_Category",
        kde=True,
        palette="Set2",
        ax=ax
    )
    ax.set_title("Distribution of AI Exposure Index")
    st.pyplot(fig)
    st.caption("Indicates how AI exposure varies among different job risk levels.")

# Education level counts (full width)
st.markdown("**Education Level Distribution**")
fig, ax = plt.subplots()
sns.countplot(
    data=df,
    x="Education_Level",
    hue="Risk_Category",
    palette="Set2",
    ax=ax
)
ax.set_title("Jobs by Education Level and Risk Category")
plt.xticks(rotation=30, ha="right")
st.pyplot(fig)
st.caption(
    "Shows how many jobs fall into each education category and how they relate to risk."
)

st.markdown("---")

# ---------------------------------------------------------------------
# Bivariate relationships
# ---------------------------------------------------------------------
st.subheader("2️⃣ Relationships Between Features and Automation Risk")

# Two-chart row
col3, col4 = st.columns(2)

with col3:
    st.markdown("**AI Exposure vs Automation Probability**")
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x="AI_Exposure_Index",
        y="Automation_Probability_2030",
        hue="Risk_Category",
        palette="Set2",
        ax=ax
    )
    ax.set_title("AI Exposure vs Automation Probability")
    st.pyplot(fig)
    st.caption("Shows whether high AI exposure increases automation risk.")

with col4:
    st.markdown("**Tech Growth vs Automation Probability**")
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x="Tech_Growth_Factor",
        y="Automation_Probability_2030",
        hue="Risk_Category",
        palette="Set2",
        ax=ax
    )
    ax.set_title("Tech Growth vs Automation Probability")
    st.pyplot(fig)
    st.caption("Highlights how technological change affects job vulnerability.")

# Boxplot: Automation by Education Level
st.markdown("**Automation Probability by Education Level**")
fig, ax = plt.subplots()
sns.boxplot(
    data=df,
    x="Education_Level",
    y="Automation_Probability_2030",
    hue="Risk_Category",
    palette="Set2",
    ax=ax
)
ax.set_title("Automation Probability by Education Level")
plt.xticks(rotation=30, ha="right")
st.pyplot(fig)
st.caption(
    "Shows how automation risk varies across education levels and risk categories."
)

st.markdown("---")

# ---------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------
st.subheader("3️⃣ Correlation Between Key Numeric Features")

corr_features = [
    "Average_Salary",
    "Years_Experience",
    "AI_Exposure_Index",
    "Tech_Growth_Factor",
    "Automation_Probability_2030"
]

corr = df[corr_features].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar_kws={"shrink": 0.7},
    ax=ax
)
ax.set_title("Correlation Heatmap")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
st.pyplot(fig)
st.caption("Shows which features most strongly correlate with automation probability.")

st.markdown("---")

# ---------------------------------------------------------------------
# Interactive Salary Filter
# ---------------------------------------------------------------------
st.subheader("4️⃣ Interactive Exploration: Salary and Automation Risk")

min_salary = int(df["Average_Salary"].min())
max_salary = int(df["Average_Salary"].max())

salary_range = st.slider(
    "Select salary range:",
    min_salary,
    max_salary,
    (min_salary, max_salary)
)

filtered_df = df[
    (df["Average_Salary"] >= salary_range[0]) &
    (df["Average_Salary"] <= salary_range[1])
]

st.write(f"Number of jobs in selected salary range: **{filtered_df.shape[0]}**")

fig, ax = plt.subplots()
sns.scatterplot(
    data=filtered_df,
    x="AI_Exposure_Index",
    y="Automation_Probability_2030",
    hue="Risk_Category",
    palette="Set2",
    ax=ax
)
ax.set_title("AI Exposure vs Automation (Filtered by Salary)")
ax.set_xlabel("AI Exposure Index")
ax.set_ylabel("Automation Probability (2030)")
st.pyplot(fig)

st.caption(
    "Lets users explore whether salary impacts the relationship between AI exposure and automation risk."
)

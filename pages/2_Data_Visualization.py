import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Data Visualization: AI Impact on Jobs (2030)")

@st.cache_data
def load_data():
    return pd.read_csv('AI_Impact_on_Jobs_2030.csv')

df = load_data()

# --- Univariate Plots ---
st.subheader("Distribution of Automation Probability (2030)")
fig, ax = plt.subplots()
sns.histplot(df["Automation_Probability_2030"], kde=True, ax=ax)
ax.set_title("Automation Probability Distribution")
ax.set_xlabel("Automation Probability")
ax.set_ylabel("Count of Jobs")
st.pyplot(fig)
st.caption("Shows whether jobs cluster in low, medium, or high automation risk ranges.")

st.subheader("Distribution of AI Exposure Index")
fig, ax = plt.subplots()
sns.histplot(df["AI_Exposure_Index"], kde=True, ax=ax)
ax.set_title("AI Exposure Index Distribution")
st.pyplot(fig)

st.subheader("Education Level Counts")
fig, ax = plt.subplots()
df["Education_Level"].value_counts().plot(kind='bar', ax=ax)
ax.set_title("Education Level Distribution")
ax.set_xlabel("Education Level")
ax.set_ylabel("Count")
plt.xticks(rotation=30)
st.pyplot(fig)

# --- Bivariate Plots ---
st.subheader("AI Exposure vs. Automation Probability")
fig, ax = plt.subplots()
sns.scatterplot(
    data=df,
    x="AI_Exposure_Index",
    y="Automation_Probability_2030",
    hue="Risk_Category",
    ax=ax
)
ax.set_title("AI Exposure vs Automation Risk")
st.pyplot(fig)

st.caption("Shows whether jobs highly exposed to AI are more likely to be automated.")

st.subheader("Tech Growth Factor vs Automation Probability")
fig, ax = plt.subplots()
sns.scatterplot(
    data=df,
    x="Tech_Growth_Factor",
    y="Automation_Probability_2030",
    hue="Risk_Category",
    ax=ax
)
ax.set_title("Tech Growth Factor vs Automation Risk")
st.pyplot(fig)

st.subheader("Automation Probability by Education Level")
fig, ax = plt.subplots()
sns.boxplot(
    data=df,
    x="Education_Level",
    y="Automation_Probability_2030",
    ax=ax
)
ax.set_title("Automation Probability by Education Level")
plt.xticks(rotation=30)
st.pyplot(fig)

# --- Correlation Heatmap ---
st.subheader("Correlation Heatmap")
corr_features = [
    "Average_Salary", "Years_Experience", "AI_Exposure_Index",
    "Tech_Growth_Factor", "Automation_Probability_2030"
]
corr = df[corr_features].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    annot_kws={'size':8},
    ax=ax
)
ax.set_title("Correlation Heatmap of Job Attributes")
plt.xticks(rotation=45)
st.pyplot(fig)

# --- Salary Filter ---
st.subheader("Interactive: Filter by Salary Range")
min_salary, max_salary = int(df['Average_Salary'].min()), int(df['Average_Salary'].max())
salary_range = st.slider('Select salary range', min_salary, max_salary, (min_salary, max_salary))
filtered_df = df[(df['Average_Salary'] >= salary_range[0]) & (df['Average_Salary'] <= salary_range[1])]

st.write(f"Jobs in selected salary range: {filtered_df.shape[0]}")

fig, ax = plt.subplots()
sns.scatterplot(
    data=filtered_df,
    x="AI_Exposure_Index",
    y="Automation_Probability_2030",
    hue="Risk_Category",
    ax=ax
)
ax.set_title("AI Exposure vs Automation Risk (Filtered)")
st.pyplot(fig)

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Data Visualization")

@st.cache_data
def load_data():
    return pd.read_csv('AI_Impact_on_Jobs_2030.csv')

df = load_data()

st.subheader("Dataset Overview")
st.write(df.describe())

# --- Univariate Plots ---
st.subheader("Distribution of Automation Probability 2030")
fig, ax = plt.subplots()
sns.histplot(df["Automation_Probability_2030"], kde=True, ax=ax)
st.pyplot(fig)

st.subheader("Distribution of AI Exposure Index")
fig, ax = plt.subplots()
sns.histplot(df["AI_Exposure_Index"], kde=True, ax=ax)
st.pyplot(fig)

st.subheader("Education Level Counts")
fig, ax = plt.subplots()
df["Education_Level"].value_counts().plot(kind='bar', ax=ax)
ax.set_xlabel("Education Level")
ax.set_ylabel("Count")
st.pyplot(fig)

# --- Bivariate Plots ---
st.subheader("AI Exposure vs. Automation Probability")
fig, ax = plt.subplots()
sns.scatterplot(x="AI_Exposure_Index", y="Automation_Probability_2030", data=df, ax=ax)
st.pyplot(fig)

st.subheader("Tech Growth Factor vs. Automation Probability")
fig, ax = plt.subplots()
sns.scatterplot(x="Tech_Growth_Factor", y="Automation_Probability_2030", data=df, ax=ax)
st.pyplot(fig)

st.subheader("Automation Probability by Education Level")
fig, ax = plt.subplots()
sns.boxplot(x="Education_Level", y="Automation_Probability_2030", data=df, ax=ax)
st.pyplot(fig)

# --- Correlation Heatmap ---
corr_features = [
    "Average_Salary", "Years_Experience", "AI_Exposure_Index",
    "Tech_Growth_Factor", "Automation_Probability_2030"
    # Add/remove columns as needed
]
corr = df[corr_features].corr()

fig, ax = plt.subplots(figsize=(8, 6))  # (width, height) in inches
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    annot_kws={'size':8},
    ax=ax,
    cbar_kws={'shrink': .7}
)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
st.pyplot(fig)

# Optional: Interactivity
st.subheader("Filter by Average Salary")
min_salary, max_salary = int(df['Average_Salary'].min()), int(df['Average_Salary'].max())
salary_range = st.slider('Select salary range', min_salary, max_salary, (min_salary, max_salary))
filtered_df = df[(df['Average_Salary'] >= salary_range[0]) & (df['Average_Salary'] <= salary_range[1])]

st.write(f"Jobs in selected salary range: {filtered_df.shape[0]}")

fig, ax = plt.subplots()
sns.scatterplot(x="AI_Exposure_Index", y="Automation_Probability_2030", data=filtered_df, ax=ax)
st.pyplot(fig)

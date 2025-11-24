import streamlit as st
import pandas as pd
import shap
from sklearn.linear_model import LinearRegression

st.title("Explainable AI: Feature Importance")

# Load data
df = pd.read_csv('AI_Impact_on_Jobs_2030.csv')

# Select relevant features for explainability
features = ['Average_Salary', 'Years_Experience', 'AI_Exposure_Index', 'Tech_Growth_Factor']
X = df[features]
y = df['Automation_Probability_2030']

# Train simple model for explanation
model = LinearRegression().fit(X, y)

# Calculate SHAP values
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Display SHAP plots
st.subheader("SHAP summary plot (feature impact)")
shap.summary_plot(shap_values, X, show=False)
st.pyplot(bbox_inches='tight')

st.markdown("""
- **AI Exposure Index** and **Tech Growth Factor** are major positive drivers for automation risk.
- Higher **Years Experience** and **Salary** may provide protective effect.
- Use these insights to understand which factors drive predictions and how individuals or businesses can adapt.
""")
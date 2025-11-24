import streamlit as st
from PIL import Image
import pandas as pd

# Load the dataset (assuming CSV format)
data = pd.read_csv('AI_Impact_on_Jobs_2030.csv')

# Display the first few rows to understand the structure
print(data.head())

# Show summary information about columns, datatype, and missing values
print(data.info())

# Show basic statistics for numerical columns
print(data.describe())
st.set_page_config(
    page_title="AI Impact on Jobs — 2030",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Impact on Jobs by the year 2030")
st.subheader("Predicting Job Automation Risk with Machine Learning")

st.markdown("""
## 🎯 Business Goal  
Companies, employees, and policymakers want to understand **which jobs are most at risk of automation** and **which skills help protect workers from AI disruption**.

Our app predicts the **probability that a job will be automated by 2030**, using job attributes such as:
- Required skills  
- Education level  
- Industry sector  
- Routine task intensity  
- Technological exposure  

These predictions help organizations:  
- Plan workforce transitions  
- Identify reskilling needs  
- Future-proof employees  
- Guide policy and education strategy  
""")

st.markdown("---")

st.markdown("""
## 📦 Dataset  
**Source:** Kaggle — *AI Impact on Jobs 2030*  
The dataset includes job-level attributes such as:
- Job Title  
- Education Requirements  
- Skills Importance  
- Routine Task Score  
- AI Automation Probability (2030 Target Variable)  

You can explore the full details in the **Data Description** page.  
""")

st.markdown("---")

st.markdown("""
## 🚀 App Structure  
Use the sidebar to navigate:

1️⃣ **Landing Page** — Business case + mission  
2️⃣ **Data Description** — Dataset structure and summary  
3️⃣ **Data Visualization** — Insights that explain patterns  
4️⃣ **Model Predictions** — Compare 2 ML models  
5️⃣ **Explainability** — SHAP analysis  
6️⃣ **Hyperparameter Tuning** — Experiment tracking (W&B)

""")

)



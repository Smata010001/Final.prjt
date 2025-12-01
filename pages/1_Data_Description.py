import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    df = pd.read_csv("StudentsPerformance.csv")
    return df

df = load_data()

st.title("Student Performance Data Description")
st.subheader("Helping schools understand how background factors shape academic outcomes")

st.markdown("""
Welcome to **Pro-Sole**, a simple tool that helps schools analyze how a student’s
background, study habits, and support systems influence their academic performance.

Our goal:  
👉 **Spot students who may need extra help**  
👉 **Identify which support programs matter most**  
👉 **Give teachers actionable insights**
""")

st.header("About the Data")
st.write("This dataset contains information about students’ backgrounds and their exam results. Below is a preview:")
st.dataframe(df.head())

st.header("Key Columns Explained")
st.markdown("""
There are 8 columns in total, covering both categorical and numerical variables. 
The categorical features include gender, race/ethnicity, parental level of education, lunch type, 
and whether the student completed a test preparation course. The numerical features include 
scores in math, reading, and writing — each ranging from 0 to 100.
""")

# ----------------------------
# Data Description
# ----------------------------
st.markdown("""
### Feature Summary
- **gender** — Student's gender (male/female)  
- **race/ethnicity** — Student’s demographic group  
- **parental level of education** — Highest education level of parents  
- **lunch** — Lunch program (standard / free-reduced)  
- **test preparation course** — Completed test-prep course (none / completed)  
- **math score** — Score in mathematics  
- **reading score** — Score in reading  
- **writing score** — Score in writing  

### What this means for your analysis
These features help schools understand how:  
- Family education level  
- Access to meal programs  
- Test preparation participation  

are connected to academic performance across subjects.
""")
# ----------------------------
# Business Case
# ----------------------------
st.header("🏫 Business Case: Why This App Matters")

st.markdown("""
Schools often struggle to understand **which background factors truly influence student
performance**. By analyzing this data, Pro-Sole helps educators:

### 🎯 Identify Students Who Need Support
See which students may fall behind based on patterns in background + scores.

### 🎯 Understand Which Programs Work
Test-prep courses, lunch programs, and family education level can impact performance — 
this app quantifies **how much**.

### 🎯 Make Data-Driven Decisions
Instead of assumptions, schools get **clear insights** to target help where it matters most.

This empowers teachers, supports students, and improves school outcomes.
""")

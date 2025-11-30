import streamlit as st
import pandas as pd
import shap
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("Explainable AI: What Drives Math Scores?")

@st.cache_data
def load_data():
    df = pd.read_csv("StudentsPerformance.csv")
    return df

df = load_data()

features = ["reading score", "writing score"]
target = "math score"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression().fit(X_train, y_train)

st.subheader("Model performance (R² on test set)")
r2 = model.score(X_test, y_test)
st.write(f"R² score: {r2:.3f}")

st.subheader("Feature importance with SHAP")

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

st.write("This summary plot shows how each feature influences the predicted math score.")

st.set_option("deprecation.showPyplotGlobalUse", False)
shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(bbox_inches="tight")

st.markdown("""
- Each point represents a student in the test set.
- The position on the x-axis shows whether that feature pushes the math score prediction up or down.
- Higher reading and writing scores generally push predicted math scores higher.
- This helps us understand how performance in one subject relates to performance in another.
""")

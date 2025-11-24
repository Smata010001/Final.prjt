import streamlit as st
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import wandb

st.title("Hyperparameter Tuning & Experiment Tracking")

# Load data
df = pd.read_csv('AI_Impact_on_Jobs_2030.csv')
features = ['Average_Salary', 'Years_Experience', 'AI_Exposure_Index', 'Tech_Growth_Factor']
X = df[features]
y = df['Automation_Probability_2030']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

st.subheader("Tune Ridge Regression alpha parameter")
alpha = st.slider("Alpha (Ridge Regularization Strength)", 0.01, 10.0, 1.0)

if st.button("Run tuning experiment"):
    wandb.init(project="jobs2030-tuning", reinit=True)
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    wandb.log({"alpha": alpha, "score": score})
    st.write(f"Model with α={alpha:.2f} achieves R²={score:.3f}")
    wandb.finish()
else:
    st.info("Adjust alpha and rerun for best model!")

st.markdown("""
Visit your W&B dashboard to compare all runs and pick the optimal model setup.
Present your results and explain what parameter choices improved the model and why.
""")
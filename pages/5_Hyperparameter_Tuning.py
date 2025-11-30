import streamlit as st
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import wandb

st.title("Hyperparameter Tuning: Ridge Regression on Math Scores")

@st.cache_data
def load_data():
    df = pd.read_csv("StudentsPerformance.csv")
    return df

df = load_data()

# Features and target
features = ["reading score", "writing score"]
target = "math score"

X = df[features]
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.subheader("Tune Ridge regularization strength (alpha)")
alpha = st.slider("Alpha (regularization strength)", 0.01, 20.0, 1.0)

st.write(
    "Higher alpha means stronger regularization, which can help reduce overfitting "
    "but might underfit if it is too large."
)

if st.button("Run tuning experiment"):
    # Start a Weights & Biases run
    wandb.init(project="students-performance-tuning", reinit=True)

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)

    # Log metrics to W&B
    wandb.log({
        "alpha": alpha,
        "r2_train": r2_train,
        "r2_test": r2_test
    })

    st.success(f"Run completed! R² train: {r2_train:.3f}, R² test: {r2_test:.3f}")
    st.write("You can compare this run with others in your Weights & Biases project dashboard.")

    wandb.finish()
else:
    st.info("Select an alpha value and click 'Run tuning experiment' to log a new run to Weights & Biases.")

st.markdown("""
- Try several alpha values to see how model performance changes.
- Look for a value where test R² is high and close to training R².
- Use the W&B dashboard to compare runs and pick the best alpha for your final model.
""")

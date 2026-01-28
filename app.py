import streamlit as st
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

# -----------------------------
# App Title & Description
# -----------------------------
st.set_page_config(page_title="Smart Loan Approval System", layout="centered")

st.title("🎯 Smart Loan Approval System – Stacking Model")
st.write(
    "This system uses a **Stacking Ensemble Machine Learning model** to predict "
    "whether a loan will be approved by combining multiple ML models for better decision making."
)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🧾 Applicant Details")

# Applicant inputs
app_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapp_income = st.sidebar.number_input("Co-Applicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Amount Term", min_value=0)

credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-Employed"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])

# -----------------------------
# Encode Inputs (Simple Encoding)
# -----------------------------
credit_history = 1 if credit_history == "Yes" else 0
employment = 1 if employment == "Self-Employed" else 0
property_map = {"Urban": 2, "Semi-Urban": 1, "Rural": 0}
property_area = property_map[property_area]

input_data = np.array([[app_income, coapp_income, loan_amount,
                        loan_term, credit_history, employment, property_area]])

# -----------------------------
# Model Architecture Display
# -----------------------------
st.subheader("🔍 Model Architecture (Stacking Ensemble)")

st.markdown("""
**Base Models Used:**
- Logistic Regression  
- Decision Tree  
- Random Forest  

**Meta Model Used:**
- Logistic Regression  

👉 Predictions from base models are combined and passed to the meta-model
to make the final decision.
""")

# -----------------------------
# Dummy Training (For Demo Purpose)
# -----------------------------
# NOTE: In real projects, models are loaded from saved files

X_dummy = np.random.rand(100, 7)
y_dummy = np.random.randint(0, 2, 100)

lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

lr.fit(X_dummy, y_dummy)
dt.fit(X_dummy, y_dummy)
rf.fit(X_dummy, y_dummy)

stack_model = StackingClassifier(
    estimators=[
        ("lr", lr),
        ("dt", dt),
        ("rf", rf)
    ],
    final_estimator=LogisticRegression()
)

stack_model.fit(X_dummy, y_dummy)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    # Base model predictions
    lr_pred = lr.predict(input_data)[0]
    dt_pred = dt.predict(input_data)[0]
    rf_pred = rf.predict(input_data)[0]

    # Stacking prediction
    final_pred = stack_model.predict(input_data)[0]

    # -----------------------------
    # Output Section
    # -----------------------------
    st.subheader("📊 Prediction Results")

    st.write("**Base Model Predictions:**")
    st.write(f"Logistic Regression → {'Approved' if lr_pred == 1 else 'Rejected'}")
    st.write(f"Decision Tree → {'Approved' if dt_pred == 1 else 'Rejected'}")
    st.write(f"Random Forest → {'Approved' if rf_pred == 1 else 'Rejected'}")

    st.subheader("🧠 Final Stacking Decision")

    if final_pred == 1:
        st.success("✅ Loan Approved")
        decision_text = "likely"
    else:
        st.error("❌ Loan Rejected")
        decision_text = "unlikely"

    # -----------------------------
    # Business Explanation
    # -----------------------------
    st.subheader("📌 Business Explanation")

    st.write(
        f"Based on income details, credit history, employment status, and combined "
        f"predictions from multiple machine learning models, the applicant is **{decision_text} "
        f"to repay the loan**. Therefore, the stacking model predicts **loan "
        f"{'approval' if final_pred == 1 else 'rejection'}**."
    )

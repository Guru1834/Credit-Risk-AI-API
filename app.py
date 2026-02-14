from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("credit_model.pkl")
columns = joblib.load("model_columns.pkl")


def preprocess(df):
    df["DTI"] = df["Loan_Repayment"] / df["Income"]

    df["Total_Expenses"] = (
        df["Rent"] + df["Groceries"] + df["Transport"] +
        df["Eating_Out"] + df["Entertainment"] +
        df["Utilities"] + df["Healthcare"] +
        df["Education"] + df["Miscellaneous"] +
        df["Insurance"]
    )

    df["Expense_Ratio"] = df["Total_Expenses"] / df["Income"]
    df["Savings_Ratio"] = df["Desired_Savings"] / df["Income"]
    df["Disposable_Income"] = df["Income"] - df["Total_Expenses"]
    df["Disposable_Ratio"] = df["Disposable_Income"] / df["Income"]

    return df


@app.get("/")
def home():
    return {"message": "Credit Score AI API Running 🚀"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])
    df = preprocess(df)

    # Fill missing one-hot columns
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    df = df[columns]

    score = float(model.predict(df)[0])

    if score < 40:
        risk_level = "High Risk"
        health = "Poor Financial Health"
    elif score < 70:
        risk_level = "Moderate Risk"
        health = "Average Financial Health"
    else:
        risk_level = "Low Risk"
        health = "Strong Financial Health"

    dti = float(df["DTI"].values[0])
    expense_ratio = float(df["Expense_Ratio"].values[0])
    savings_ratio = float(df["Savings_Ratio"].values[0])
    disposable_ratio = float(df["Disposable_Ratio"].values[0])

    recommendations = []

    if dti > 0.5:
        recommendations.append("Your debt-to-income ratio is high. Consider reducing loan burden.")

    if savings_ratio < 0.2:
        recommendations.append("Try increasing your savings percentage for better financial stability.")

    if expense_ratio > 0.7:
        recommendations.append("Your expenses are consuming a large portion of your income. Review discretionary spending.")

    if disposable_ratio < 0.2:
        recommendations.append("Low disposable income detected. Improve income-to-expense balance.")

    if not recommendations:
        recommendations.append("Your financial profile looks stable. Maintain your current financial discipline.")

    return {
        "credit_score": round(score, 2),
        "risk_level": risk_level,
        "financial_health": health,
        "metrics": {
            "dti": round(dti, 2),
            "expense_ratio": round(expense_ratio, 2),
            "savings_ratio": round(savings_ratio, 2),
            "disposable_ratio": round(disposable_ratio, 2)
        },
        "recommendations": recommendations
    }

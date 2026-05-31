from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import numpy as np
import sys

from custom_logistic_regression import CustomLogisticRegression

regr = CustomLogisticRegression()


app = FastAPI(
    title="Credit Risk Prediction API",
    description="An API to predict the likelihood of a credit card holder defaulting on their payment based on various features.",
    version="1.0.0"
)

class ApplicantData(BaseModel):
    limit_bal: float = Field(..., description='Credit card limit balance of the applicant', examples=[50000.00])
    sex: int = Field(..., description='Gender of the applicant (1 = Male, 2 = Female)', examples=[1])
    education: int = Field(..., description='Education level of the applicant (1 = Graduate school, 2 = University, 3 = High school, 4 = Others)', examples=[2])
    marriage: int = Field(..., description='Marriage status of the applicant (1 = Married, 2 = Single, 3 = Others)', examples=[2])
    age: int = Field(..., description='Age of the applicant', examples=[30])
    utilisation_rate: float = Field(..., description='Utilisation rate of the applicant', examples=[0.15])
    pay_to_bill_ratio: float = Field(..., description='Pay to bill ratio of the applicant', examples=[0.85])
    delinquency_score: float = Field(..., description='Delinquency score of the applicant', examples=[0.2])

class ScorecardTransformer:
    def __init__(self, pdo=20,baseline_score=600, baseline_odds=2):
        self.factor = pdo / np.log(2)
        self.offset = baseline_score - self.factor * np.log(baseline_odds)

    def transform(self, probability):
        probabilities = np.clip(probability, 1e-5, 1 - 1e-5)
        odds = (1 - probabilities) / probabilities
        score = self.offset + self.factor * np.log(odds)

        return int(np.clip(np.round(score), 300, 850))  # Ensure scores are within the range of 300 to 850

# Load the model and training data statistics at startup
try:
    import pickle
    with open('credit_risk_model.pkl', 'rb') as f:
        model_package = pickle.load(f)

        model = model_package['model']
        training_mean = model_package['training_mean']
        training_std = model_package['training_std']

except Exception as e:
    print(f"Error loading model: {e}")
    # we exit abruptly since the model is essential for the API to function
    sys.exit(1)

@app.post("/predict", response_model=Dict[str, Any])
def predict(applicant: ApplicantData):
    try:
        raw_features = np.array([applicant.limit_bal,
                                applicant.sex,
                                applicant.education,
                                applicant.marriage,
                                applicant.age,
                                applicant.utilisation_rate,
                                applicant.pay_to_bill_ratio,
                                applicant.delinquency_score]).reshape(1, -1)
        
        scaled_features = (raw_features - training_mean) / training_std

        prediction = float(model._predict_proba(scaled_features)[0][1])  # probability of defaulting

        scorecard_transformer = ScorecardTransformer()
        credit_score = scorecard_transformer.transform(prediction)

        OPTIMAL_THRESHOLD = 0.11 # as per the model evaluation, we set the threshold at 11% probability of defaulting

        risk_category = ''
        decision = ''
        print(f"Prediction: {prediction}, Credit Score: {credit_score}")  # Debugging statement to check the prediction and credit score values
    
        if prediction >= OPTIMAL_THRESHOLD and prediction <= 0.15:
            decision = 'Manual Review'
            risk_category = "Medium Risk"
        elif prediction >= OPTIMAL_THRESHOLD:
            decision = 'Reject'
            risk_category = "High Risk"
        else:
            decision = 'Approve'
            risk_category = "Low Risk"
        
        return {
            "credit_score": credit_score,
            'defaulting_probability': round(prediction, 4),
            'risk_category': risk_category,
            'decision': decision
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'An error occurred during prediction: {str(e)}')

from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
from pydantic_models import CustomerData, PredictionResponse

app = FastAPI()

# Load the best model
model = mlflow.sklearn.load_model("models:/CreditRiskModel/Production")

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: CustomerData):
    input_df = pd第二次
    pd.DataFrame([data.dict()])
    input_df = input_df.drop(['CustomerId'], axis=1)
    prob = model.predict_proba(input_df)[0][1]
    return PredictionResponse(risk_probability=prob)
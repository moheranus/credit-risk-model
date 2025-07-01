from fastapi import FastAPI
import mlflow
import mlflow.sklearn
import pandas as pd
from pydantic_models import CustomerData, PredictionResponse

app = FastAPI()

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Load the best model
model = mlflow.sklearn.load_model("models:/CreditRiskModel/Production")

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: CustomerData):
    input_df = pd.DataFrame([data.dict()])
    input_df = input_df.drop(['CustomerId'], axis=1)
    prob = model.predict_proba(input_df)[:, 1][0]
    return PredictionResponse(risk_probability=prob)
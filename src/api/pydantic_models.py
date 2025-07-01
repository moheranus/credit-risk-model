from pydantic import BaseModel
from typing import List

class CustomerData(BaseModel):
    CustomerId: str
    Amount: float
    Value: float
    TransactionHour: int
    TransactionDay: int
    TransactionMonth: int
    TotalAmount: float
    AvgAmount: float
    StdAmount: float
    ProductCategory_airtime: float
    ProductCategory_financial_services: float
    # Add other one-hot encoded features as needed

class PredictionResponse(BaseModel):
    risk_probability: float
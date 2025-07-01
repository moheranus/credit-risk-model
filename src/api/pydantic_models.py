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
    ProductCategory_data_bundles: float
    ProductCategory_financial_services: float
    ProductCategory_movies: float
    ProductCategory_other: float
    ProductCategory_ticket: float
    ProductCategory_transport: float
    ProductCategory_tv: float
    ProductCategory_utility_bill: float
    ChannelId_ChannelId_1: float
    ChannelId_ChannelId_2: float
    ChannelId_ChannelId_3: float
    ChannelId_ChannelId_5: float
    PricingStrategy_0: float
    PricingStrategy_1: float
    PricingStrategy_2: float
    PricingStrategy_4: float

class PredictionResponse(BaseModel):
    risk_probability: float
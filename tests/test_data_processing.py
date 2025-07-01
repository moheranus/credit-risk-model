import pytest
import pandas as pd
from src.data_processing import create_rfm_features, extract_time_features

def test_create_rfm_features():
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'TransactionId': ['T1', 'T2', 'T3'],
        'Amount': [100, 200, 150],
        'TransactionStartTime': ['2023-01-01', '2023-01-02', '2023-01-03']
    })
    rfm = create_rfm_features(df)
    assert set(rfm.columns) == {'CustomerId', 'Recency', 'Frequency', 'Monetary'}
    assert rfm.shape[0] == 2  # Two customers

def test_extract_time_features():
    df = pd.DataFrame({
        'TransactionStartTime': ['2023-01-01 14:30:00']
    })
    df = extract_time_features(df)
    assert set(['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']).issubset(df.columns)
    assert df['TransactionHour'][0] == 14
    assert df['TransactionDay'][0] == 1
    assert df['TransactionMonth'][0] == 1
    assert df['TransactionYear'][0] == 2023
import sys
import os
import pandas as pd
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_processing import create_rfm_features, extract_time_features

def test_create_rfm_features():
    # Create a sample DataFrame
    data = {
        'CustomerId': ['C1', 'C1', 'C2'],
        'TransactionStartTime': ['2023-01-01 10:00:00', '2023-01-02 10:00:00', '2023-01-03 10:00:00'],
        'TransactionId': [1, 2, 3],
        'Amount': [100, 200, 300]
    }
    df = pd.DataFrame(data)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    result = create_rfm_features(df)
    assert 'Recency' in result.columns
    assert 'Frequency' in result.columns
    assert 'Monetary' in result.columns
    assert len(result) == 2  # Two unique CustomerIds

def test_extract_time_features():
    # Create a sample DataFrame
    data = {
        'CustomerId': ['C1'],
        'TransactionStartTime': ['2023-01-01 10:15:30']
    }
    df = pd.DataFrame(data)
    
    result = extract_time_features(df)
    assert 'TransactionHour' in result.columns
    assert 'TransactionDay' in result.columns
    assert 'TransactionMonth' in result.columns
    assert 'TransactionYear' in result.columns
    assert result['TransactionHour'].iloc[0] == 10
    assert result['TransactionDay'].iloc[0] == 1
    assert result['TransactionMonth'].iloc[0] == 1
    assert result['TransactionYear'].iloc[0] == 2023
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from xverse.transformer import WOE

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def create_rfm_features(df):
    """Calculate RFM metrics."""
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,  # Recency
        'TransactionId': 'count',  # Frequency
        'Amount': 'sum'  # Monetary
    }).reset_index()
    rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
    return rfm

def cluster_customers(rfm):
    """Cluster customers using K-Means."""
    rfm_scaled = StandardScaler().fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Identify high-risk cluster (low frequency, low monetary)
    cluster_stats = rfm.groupby('Cluster').agg({
        'Frequency': 'mean',
        'Monetary': 'mean'
    })
    high_risk_cluster = cluster_stats.idxmin().iloc[0]  # Cluster with lowest frequency
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    return rfm[['CustomerId', 'is_high_risk']]

def create_aggregate_features(df):
    """Create other aggregate features."""
    agg_features = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'count', 'std'],
        'TransactionStartTime': ['min', 'max']
    }).reset_index()
    agg_features.columns = [
        'CustomerId', 'TotalAmount', 'AvgAmount', 'TransactionCount', 'StdAmount',
        'FirstTransaction', 'LastTransaction'
    ]
    return agg_features

def extract_time_features(df):
    """Extract time-based features."""
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year
    return df

def build_pipeline():
    """Build a sklearn pipeline for preprocessing."""
    numerical_features = ['Amount', 'Value', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TotalAmount', 'AvgAmount', 'StdAmount']
    categorical_features = ['ProductCategory', 'ChannelId', 'PricingStrategy']
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('woe', WOE())  # Requires is_high_risk
    ])
    
    return pipeline

def process_data(input_path, output_path):
    """Process raw data and save to processed directory."""
    df = load_data(input_path)
    df = extract_time_features(df)
    agg_df = create_aggregate_features(df)
    rfm = create_rfm_features(df)
    rfm_with_risk = cluster_customers(rfm)
    
    df = df.merge(agg_df, on='CustomerId', how='left')
    df = df.merge(rfm_with_risk, on='CustomerId', how='left')
    
    pipeline = build_pipeline()
    processed_data = pipeline.fit_transform(df, woe__y=df['is_high_risk'])
    
    # Convert back to DataFrame for saving
    feature_names = (numerical_features + 
                     pipeline.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_features).tolist())
    processed_df = pd.DataFrame(processed_data, columns=feature_names)
    processed_df['CustomerId'] = df['CustomerId'].reset_index(drop=True)
    processed_df['is_high_risk'] = df['is_high_risk'].reset_index(drop=True)
    
    processed_df.to_csv(output_path, index=False)
    return processed_df

if __name__ == "__main__":
    input_path = 'data/raw/xente_data.csv'
    output_path = 'data/processed/processed_data.csv'
    process_data(input_path, output_path)
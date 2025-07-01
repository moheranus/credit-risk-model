import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from mlflow.models.signature import infer_signature

def load_processed_data(file_path):
    return pd.read_csv(file_path)

def train_model():
    df = load_processed_data('data/processed/processed_data.csv')
    X = df.drop(['CustomerId', 'is_high_risk'], axis=1)
    y = df['is_high_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'LogisticRegression': LogisticRegression(),
        'GradientBoosting': GradientBoostingClassifier()
    }
    
    param_grids = {
        'LogisticRegression': {'C': [0.1, 1, 10]},
        'GradientBoosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
    }
    
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            grid = GridSearchCV(model, param_grids[name], cv=5, scoring='roc_auc')
            grid.fit(X_train, y_train)
            
            y_pred = grid.predict(X_test)
            y_prob = grid.predict_proba(X_test)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_prob)
            }
            
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metrics)
            
            # Infer model signature
            signature = infer_signature(X_train, grid.predict(X_train))
            mlflow.sklearn.log_model(grid.best_estimator_, artifact_path=f"{name}_model", signature=signature)
            
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model = grid.best_estimator_
    
    # Register the best model
    signature = infer_signature(X_train, best_model.predict(X_train))
    mlflow.sklearn.log_model(best_model, artifact_path="best_model", registered_model_name="CreditRiskModel", signature=signature)

if __name__ == "__main__":
    train_model()
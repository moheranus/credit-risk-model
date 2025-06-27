Credit Risk Model for Bati Bank

This repository implements a Credit Risk Probability Model for Bati Bank, in collaboration with an eCommerce platform to enable a buy-now-pay-later service. The model uses Recency, Frequency, and Monetary (RFM) data to predict credit risk, assign credit scores, and determine optimal loan terms.

Project Structure

credit-risk-model/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 1.0-eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/
│   └── test_data_processing.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md

Credit Scoring Business Understanding
How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?
The Basel II Capital Accord mandates rigorous risk measurement and management to ensure financial institutions hold sufficient capital against credit risks. It emphasizes transparency under Pillar 2 (supervisory review) and Pillar 3 (market discipline), requiring models to be interpretable and well-documented. For Bati Bank, an interpretable model like Logistic Regression with Weight of Evidence (WoE) enables stakeholders to understand how RFM-based features contribute to risk scores, facilitating regulatory audits and compliance. Thorough documentation ensures traceability and reproducibility, aligning with Basel II’s standards for validating credit risk models.
Why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?
Without a direct "default" label in the dataset, a proxy variable (is_high_risk) is essential to classify customers as high or low risk. This proxy is derived by clustering customers based on RFM metrics (e.g., using K-Means) and identifying the least engaged group (low frequency, low monetary value) as high-risk, presuming disengagement correlates with default likelihood. However, this approach risks misclassification: false positives may deny credit to viable customers, reducing revenue, while false negatives could approve risky loans, increasing defaults. These errors may harm Bati Bank’s profitability and reputation, necessitating careful validation of the proxy.
What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?
Simple models like Logistic Regression with WoE are interpretable, making it easier to explain predictions to regulators and customers, which is critical for Basel II compliance. They are computationally efficient but may underfit complex data patterns, leading to lower predictive accuracy. Complex models like Gradient Boosting capture non-linear relationships, offering higher accuracy for RFM-based risk prediction. However, their opacity complicates regulatory validation and increases scrutiny. For Bati Bank, the trade-off involves balancing interpretability and compliance with predictive performance, with simple models often preferred for transparency in regulated environments.
Setup Instructions

Clone the repository:git clone https://github.com/your-username/credit-risk-model.git
cd credit-risk-model


Install dependencies:pip install -r requirements.txt


Place raw data in data/raw/ (add to .gitignore).
Run EDA notebook:jupyter notebook notebooks/1.0-eda.ipynb


Process data:python src/data_processing.py


Train model:python src/train.py


Start API:uvicorn src.api.main:app --reload


Run tests:pytest tests/



Dependencies

Python 3.8+
pandas
numpy
scikit-learn
mlflow
pytest
fastapi
uvicorn
flake8
xverse
woe

See requirements.txt for details.
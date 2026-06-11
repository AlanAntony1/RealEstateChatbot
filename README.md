# Real Estate Financial Assistant

A full-stack AI-powered financial assistant for a real estate company, built with Vertex AI ADK, AWS SageMaker, PostgreSQL, and Streamlit.

## Live Demo

[Streamlit App](https://realestatechatbot-tmtvghtrfxkiqzbvamd8bu.streamlit.app)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                          │
│  Chatbot | Property Data | Financials | Press Releases  │
│              ML Predictions                             │
└────────┬───────────────────────────┬────────────────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐        ┌──────────────────────┐
│  Vertex AI ADK  │        │   AWS SageMaker       │
│  Gemini 2.5     │        │  Regression Endpoint  │
│  Flash Chatbot  │        │  Classification       │
│  4 Tools        │        │  Endpoint             │
└────────┬────────┘        └──────────────────────┘
         │
    ┌────┴─────────────────────┐
    │                          │
    ▼                          ▼
┌──────────┐          ┌───────────────┐
│PostgreSQL│          │  SEC EDGAR    │
│Properties│          │  Prologis     │
│Financials│          │  10-K / 10-Q  │
└──────────┘          └───────────────┘
         │
         ▼
┌──────────────────┐
│  Press Releases  │
│  press_releases  │
│  .json           │
└──────────────────┘
```

---

## Features

- **Conversational Chatbot** — Vertex AI ADK agent powered by Gemini 2.5 Flash that routes queries to the appropriate data source
- **Property Database** — PostgreSQL with 25 properties across 5 metro areas with full financial data
- **Financial Summary** — Aggregated revenue, net income, and expenses by metro area with charts
- **Press Releases** — 8 mock press releases covering acquisitions, earnings, expansions, and more
- **SEC EDGAR Integration** — Real Prologis (PLD) financial data from SEC filings
- **ML Predictions** — Two SageMaker-hosted models for regression and classification
- **Multi-cloud** — AWS SageMaker (ML endpoints) + GCP Vertex AI (chatbot)

---

## Data Sources

| Source | Description |
|---|---|
| PostgreSQL | 25 properties across Chicago, New York, Dallas, Seattle, Atlanta |
| SEC EDGAR | Prologis (PLD) annual and quarterly filings |
| Press Releases | 8 mock releases (earnings, acquisitions, expansions) |
| California Housing | sklearn dataset for regression training |
| Bank Marketing | UCI dataset for classification training |

---

## ML Models

### Regression — California Housing
- **Model**: Random Forest Regressor
- **Task**: Predict median house value
- **Metrics**: RMSE, MAE, R²
- **Deployment**: AWS SageMaker endpoint `realestate-regression`

### Classification — Bank Marketing
- **Model**: Logistic Regression
- **Task**: Predict customer subscription (yes/no)
- **Metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix
- **Deployment**: AWS SageMaker endpoint `realestate-classification`

---

## Chatbot Query Routing

The Vertex AI ADK agent uses 4 tools to route queries:

| Tool | Trigger | Data Source |
|---|---|---|
| `query_properties` | Property questions, metro area, type filters | PostgreSQL |
| `get_financial_summary` | Revenue totals, comparisons across regions | PostgreSQL |
| `search_press_releases` | News, acquisitions, announcements | press_releases.json |
| `get_sec_financials` | Quarterly/annual SEC filings | edgar_data.json |

---

## Project Structure

```
InternshipProject/
├── streamlit_app.py              # Main Streamlit UI
├── chatbot_agent.py              # Vertex AI ADK chatbot (standalone)
├── mock_data.py                  # Mock data fallback for public demo
├── press_releases.json           # 8 mock press releases
├── edgar_data.json               # SEC EDGAR Prologis data
├── sec_edgar.py                  # Script to fetch EDGAR data
├── deploy_sagemaker.py           # SageMaker deployment script
├── inference_scripts/
│   └── inference.py              # Regression inference script
├── inference_scripts_clf/
│   └── inference.py              # Classification inference script
├── model_artifacts/              # Trained model files (not in git)
│   ├── rf_regressor.joblib
│   ├── scaler_regression.joblib
│   ├── logistic_classifier.joblib
│   ├── scaler_classification.joblib
│   ├── label_encoders.joblib
│   └── feature_columns.joblib
├── regression_california_housing.ipynb   # Training notebook
├── classification_bank_marketing.ipynb   # Training notebook
├── schema.sql                    # PostgreSQL schema and seed data
├── requirements.txt
└── README.md
```

---

## Local Setup

### Prerequisites
- Python 3.8+
- PostgreSQL
- AWS account with SageMaker endpoints deployed
- GCP account with Vertex AI enabled

### 1. Clone the repo
```bash
git clone https://github.com/AlanAntony1/RealEstateChatbot.git
cd RealEstateChatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up PostgreSQL
```bash
createdb realestate_db
psql -d realestate_db -f schema.sql
```

### 4. Fetch SEC EDGAR data
```bash
python sec_edgar.py
```

### 5. Train ML models
Open and run both notebooks:
```bash
jupyter notebook
```
- `regression_california_housing.ipynb`
- `classification_bank_marketing.ipynb`

### 6. Deploy to SageMaker
```bash
python deploy_sagemaker.py
```

### 7. Set environment variables
```bash
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=realestate-assistant-499113
export GOOGLE_CLOUD_LOCATION=us-central1
export AWS_DEFAULT_REGION=us-east-1
```

### 8. Run the app
```bash
streamlit run streamlit_app.py
```

---

## Cloud Setup

### GCP — Vertex AI ADK
1. Create a GCP project at https://console.cloud.google.com
2. Enable the Vertex AI API
3. Authenticate: `gcloud auth application-default login`
4. The chatbot uses `gemini-2.5-flash` via Vertex AI

### AWS — SageMaker
1. Create an AWS account and configure credentials: `aws configure`
2. Create an S3 bucket for model artifacts
3. Create a SageMaker execution role with S3 and SageMaker permissions
4. Run `deploy_sagemaker.py` to deploy both endpoints

### Streamlit Cloud
1. Push code to GitHub
2. Go to https://share.streamlit.io and connect your repo
3. Add secrets in the app settings:
```toml
GOOGLE_API_KEY = "your_gemini_api_key"
AWS_ACCESS_KEY_ID = "your_aws_key"
AWS_SECRET_ACCESS_KEY = "your_aws_secret"
AWS_DEFAULT_REGION = "us-east-1"
```

---

## Multi-Cloud Integration

| Cloud | Service | Usage |
|---|---|---|
| GCP | Vertex AI ADK | Chatbot agent with Gemini 2.5 Flash |
| AWS | SageMaker | Hosted ML inference endpoints |
| AWS | S3 | Model artifact storage |

---

## Tech Stack

- **Frontend**: Streamlit
- **Chatbot**: Google Vertex AI ADK, Gemini 2.5 Flash
- **ML**: scikit-learn, AWS SageMaker
- **Database**: PostgreSQL, psycopg2
- **Cloud**: GCP Vertex AI, AWS SageMaker, AWS S3
- **Data**: SEC EDGAR API, UCI Bank Marketing, California Housing

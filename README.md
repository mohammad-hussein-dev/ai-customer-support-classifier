# AI Customer Support Ticket Classifier

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> An end-to-end machine learning pipeline for automated classification of customer support tickets using TF-IDF and Logistic Regression.

## Features

- **4 Categories**: Billing, Technical Support, Account, Refund
- **5,000 Synthetic Tickets** with realistic class imbalance
- **Complete ML Pipeline**: Preprocessing -> Feature Engineering -> Training -> Evaluation -> Deployment
- **Production-Ready**: Confidence thresholding with human-in-the-loop routing
- **Interactive App**: Streamlit web interface for real-time classification
- **Comprehensive EDA**: 20+ publication-quality visualizations
- **Cross-Validation**: Stratified 5-fold CV for reliable metrics

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd ai-customer-support-classifier
python3 -m venv venv --prompt=ticket-classifier
source venv/bin/activate
pip install -r requirements.txt

# Run full pipeline
python notebooks/exploratory/train_and_evaluate.py

# Launch Streamlit app
streamlit run deployment/app.py
```

## Project Structure

```
ai-customer-support-classifier/
├── config/                 # YAML configuration files
├── data/
│   ├── raw/               # Original datasets
│   └── processed/         # Cleaned datasets
├── deployment/
│   └── app.py             # Streamlit application
├── models/
│   ├── baseline/          # Experimental models
│   └── production/        # Deployed artifacts
├── notebooks/
│   └── exploratory/       # EDA and experiments
├── reports/
│   ├── figures/           # Visualizations
│   └── report.pdf         # LaTeX report
├── src/
│   ├── data/              # Preprocessing
│   ├── features/          # Feature engineering
│   ├── models/            # Training & evaluation
│   └── visualization/     # Plotting utilities
└── tests/                 # Unit & integration tests
```

## Model Performance

| Model | Accuracy | F1 (Weighted) | F1 (Macro) |
|-------|----------|---------------|------------|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 |
| Linear SVM | 1.0000 | 1.0000 | 1.0000 |
| Naive Bayes | 1.0000 | 1.0000 | 1.0000 |

*Note: Perfect scores reflect the synthetic nature of the dataset. Real-world deployment would use more diverse data.*

## API Usage

```python
from src.models.predict_model import TicketClassifier

clf = TicketClassifier.from_directory("models/production")
result = clf.predict("I was charged twice this month.")

print(result["category"])      # Billing
print(result["confidence"])    # 0.9330
print(result["needs_review"])  # False
```

## Configuration

All hyperparameters are centralized in `config/config.yaml`:

```yaml
features:
  vectorizer: "tfidf"
  max_features: 10000
  ngram_range: [1, 2]

model:
  name: "logistic_regression"
  class_weight: "balanced"
  C: 1.0
```

## Docker

```bash
docker-compose up --build
# App available at http://localhost:8501
```

## License

MIT License - see [LICENSE](LICENSE) file.
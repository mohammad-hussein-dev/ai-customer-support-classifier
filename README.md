# 🤖 AI Customer Support Ticket Classifier v1.0

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)](.github/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Production-ready NLP pipeline** for automated customer support ticket classification.
> Trained on **13,783 real banking tickets** (Banking77) + synthetic data with realistic noise injection.
> Deployed via FastAPI with auto-generated Swagger documentation.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Live API](#-live-api)
- [What It Does](#-what-it-does)
- [Performance](#-performance)
- [Architecture](#-architecture)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Testing](#-testing)
- [Development Workflow](#-development-workflow)
- [Error Analysis](#-error-analysis)
- [Future Roadmap](#-future-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 🎯 Overview

Customer support teams receive thousands of tickets daily. Manual triage causes:

- **Delays:** Average 4-6 hours response time
- **Errors:** 15-20% misclassification rate
- **Cost:** $2.50 per manual ticket routing
- **Scale:** Cannot handle peak loads

This project solves these problems with an end-to-end ML pipeline that:

1. **Classifies** tickets into 4 categories automatically
2. **Explains** why each prediction was made
3. **Monitors** model health and performance
4. **Deploys** as a production REST API

---

## 🌐 Live API

```
http://localhost:8000/docs
```

Interactive Swagger UI with auto-generated documentation.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single ticket classification |
| `/predict/batch` | POST | Batch classification (max 100) |
| `/explain` | POST | Explain prediction (top words) |
| `/explain` | POST | Explain prediction (top words) |
| `/health` | GET | Model status & metrics |

---

## 🎬 What It Does

Customer sends a ticket → AI classifies it automatically:

| Input | Output | Confidence |
|-------|--------|------------|
| "I was charged twice for my subscription" | **Billing** | 99.3% |
| "Can't login to my account anymore" | **Account** | 97.1% |
| "App keeps crashing after update" | **Technical Support** | 94.2% |
| "I want my money back please" | **Refund** | 98.7% |

---

## 📊 Performance

### Model Comparison (5-Fold Stratified CV + Test)

| Rank | Model | CV F1 | Test F1 | Accuracy | Model Size |
|------|-------|-------|---------|----------|------------|
| 🥇 | **SVM (RBF)** | 0.9094 | **0.9229** | **92.3%** | 2.1 MB |
| 🥈 | Logistic Regression | 0.8879 | 0.8947 | 89.6% | 45 KB |
| 🥉 | Random Forest | 0.8209 | 0.8409 | 84.6% | 12 KB |

### Per-Class F1 (Best Model: SVM)

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Account | 0.93 | 0.95 | 0.94 | 2,134 |
| Billing | 0.91 | 0.93 | 0.92 | 1,222 |
| Technical Support | 0.89 | 0.87 | 0.88 | 508 |
| Refund | 0.85 | 0.82 | 0.83 | 273 |
| **Macro Avg** | **0.90** | **0.89** | **0.89** | 4,135 |

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 13,783 |
| Real (Banking77) | 13,083 |
| Synthetic (Refund boost) | 700 |
| Features (TF-IDF) | 3,000 |
| Train / Test Split | 70% / 30% (Stratified) |
| Class Imbalance | Refund: 6.6% |

---

## 🏗️ Architecture

```
Raw Ticket
    |
    v
TextPreprocessor (NLTK)
    - Lowercase
    - Regex cleaning
    - Tokenization
    - Lemmatization
    - Stopword removal
    |
    v
TF-IDF Vectorizer (scikit-learn)
    - N-gram: (1, 2)
    - Max features: 3,000
    - Sublinear TF
    - Min DF: 3 | Max DF: 0.90
    |
    v
SVM Classifier (RBF Kernel)
    - C = 1.0
    - class_weight: balanced
    - 4-class output
    |
    v
Category + Confidence + All Probabilities
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| ✅ **Real Data** | 13K Banking77 tickets + 700 synthetic with realistic noise |
| ✅ **Stratified Split** | Preserves class imbalance (Refund: 6.6%) |
| ✅ **Class Balancing** | `class_weight="balanced"` for fairness |
| ✅ **Cross-Validation** | 5-Fold Stratified CV with statistical significance |
| ✅ **Error Analysis** | 320 misclassifications analyzed with business insights |
| ✅ **Explainability** | TF-IDF word importance for every prediction |
| ✅ **Production API** | FastAPI with auto-generated Swagger docs |
| ✅ **Type Safety** | Full Pydantic validation on all endpoints |
| ✅ **CI/CD** | GitHub Actions with automated testing |
| ✅ **GitLab Mirror** | Auto-sync via SSH deploy key |

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **Languages** | Python 3.10+ |
| **ML / NLP** | scikit-learn, NLTK, NumPy, Pandas |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Data** | Faker, urllib (built-in) |
| **Visualization** | Matplotlib |
| **Testing** | pytest, unittest |
| **CI/CD** | GitHub Actions |
| **Containerization** | Docker, Docker Compose |
| **Documentation** | LaTeX, Beamer, TikZ |
| **Development OS** | Arch Linux |

---

## 🚀 Installation

### Option 1: Docker

```bash
git clone https://github.com/mohammad-hussein-dev/ai-customer-support-classifier.git
cd ai-customer-support-classifier
docker-compose up -d
```

### Option 2: Manual Setup

```bash
# Clone repository
git clone https://github.com/mohammad-hussein-dev/ai-customer-support-classifier.git
cd ai-customer-support-classifier

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Requirements

```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
nltk>=3.8.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
matplotlib>=3.7.0
```

---

## 📖 Usage

### 1. Build Dataset

```bash
python3 scripts/build_hybrid_dataset.py   # Downloads 13K real tickets
python3 scripts/preprocess_and_split.py   # 70/30 stratified split
```

### 2. Train & Evaluate

```bash
python3 scripts/train_and_evaluate.py     # Trains 3 models, picks best
python3 scripts/error_analysis.py         # Error patterns + insights
```

### 3. Run API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Open: http://localhost:8000/docs
```

### Example API Call

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I was charged twice for my subscription"}'
```

**Response:**

```json
{
  "category": "Billing",
  "confidence": 0.9929,
  "all_probabilities": {
    "Account": 0.0047,
    "Billing": 0.9929,
    "Refund": 0.0003,
    "Technical Support": 0.0022
  },
  "model_version": "v1.0",
  "timestamp": "2026-08-21T10:00:00"
}
```

---

## 📘 API Documentation

Auto-generated Swagger UI available at:

```
http://localhost:8000/docs
```

### Endpoints

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/predict` | POST | `{"text": "..."}` | Category + Confidence + Probabilities |
| `/predict/batch` | POST | `{"texts": ["...", "..."]}` | List of predictions |
| `/explain` | POST | `{"text": "..."}` | Top contributing words |
| `/health` | GET | — | Model status, F1, Accuracy |

---

## 📁 Project Structure

```
ai-customer-support-classifier/
│
├── api/                          # FastAPI production API
│   └── main.py                   # /predict, /explain, /health endpoints
│
├── scripts/                      # Pipeline automation
│   ├── build_hybrid_dataset.py   # Download Banking77 + synthetic merge
│   ├── preprocess_and_split.py   # Text cleaning + stratified split
│   ├── train_and_evaluate.py     # 3-model training + CV + selection
│   └── error_analysis.py         # Misclassification analysis + insights
│
├── src/                          # Core ML modules
│   ├── data/
│   │   ├── make_dataset.py       # Synthetic data generator
│   │   └── preprocessing.py      # TextPreprocessor (NLTK pipeline)
│   ├── features/
│   │   └── build_features.py     # TF-IDF vectorizer
│   ├── models/
│   │   ├── train_model.py        # Model training + GridSearchCV
│   │   ├── evaluate_model.py     # Classification report + confusion matrix
│   │   └── predict_model.py      # TicketClassifier production class
│   ├── utils/
│   │   ├── config.py             # YAML configuration loader
│   │   ├── logger.py             # Structured logging
│   │   ├── metrics.py            # Custom metric functions
│   │   └── helpers.py            # Utility functions
│   └── visualization/
│       └── visualize.py          # EDA plots and dashboards
│
├── models/                       # Serialized models
│   ├── baseline/                 # LR, SVM, NB baseline models
│   └── production_v2/            # Best model + vectorizer + encoder
│       ├── model.pkl
│       ├── vectorizer.pkl
│       ├── encoder.pkl
│       └── metrics.json
│
├── data/                         # Datasets
│   ├── raw/
│   │   ├── tickets.csv           # Original synthetic data
│   │   └── hybrid_dataset.csv    # Real + synthetic merged
│   └── processed/
│       ├── train.csv
│       ├── test.csv
│       ├── y_train.npy
│       └── y_test.npy
│
├── reports/                      # Analysis outputs
│   ├── figures/                  # 30+ EDA visualizations
│   │   ├── 00_comprehensive_dashboard.png
│   │   ├── confusion_matrix.png
│   │   ├── confidence_distribution.png
│   │   └── shap_summary.png
│   ├── tables/
│   │   ├── model_comparison.csv
│   │   ├── eda_summary.csv
│   │   └── error_analysis.json
│   ├── report.tex                # 17-page LaTeX technical report
│   └── report.pdf
│
├── notebooks/                    # Jupyter experiments
│   └── exploratory/
│       ├── run_eda.py
│       ├── test_pipeline.py
│       └── train_and_evaluate.py
│
├── tests/                        # Automated tests
│   ├── unit/
│   │   ├── test_preprocessing.py
│   │   ├── test_features.py
│   │   └── test_model.py
│   └── integration/
│       └── test_pipeline.py
│
├── .github/workflows/            # CI/CD pipelines
│   ├── ci.yml                    # pytest + flake8 + mypy
│   ├── cd.yml                    # Docker build on release
│   └── mirror-to-gitlab.yml      # Auto-sync to GitLab
│
├── deployment/                   # Streamlit (legacy v1.0)
│   ├── app.py
│   └── Dockerfile
│
├── config/                       # YAML configurations
│   ├── config.yaml
│   ├── logging.yaml
│   └── model_params.yaml
│
├── Dockerfile                    # Production container
├── docker-compose.yml            # Multi-service orchestration
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
├── pyproject.toml                # Project metadata
├── Makefile                      # Common commands
└── README.md                     # This file
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Open coverage report
# htmlcov/index.html
```

**Current Coverage:** 95%+

| Module | Coverage |
|--------|----------|
| `preprocessing.py` | 98% |
| `build_features.py` | 95% |
| `train_model.py` | 92% |
| `predict_model.py` | 96% |

---

## 🔄 Development Workflow

### Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Production branch |
| `develop` | Integration branch |
| `feature/*` | New features |
| `fix/*` | Bug fixes |

### Commit Convention

Following [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(models): add Random Forest classifier
fix(api): resolve empty text validation
docs(readme): update performance metrics
test(pipeline): add end-to-end integration test
```

### Code Quality

| Tool | Purpose |
|------|---------|
| Black | Code formatting |
| Ruff | Linting |
| MyPy | Type checking |

---

## 🔍 Error Analysis

**Total misclassified:** 320 / 4,135 (7.7%)

### Top Confused Pairs

| True → Predicted | Count | Insight |
|------------------|-------|---------|
| Account → Billing | 94 | Boundary words: "card", "payment" |
| Billing → Account | 76 | Shared vocabulary overlap |
| Account → Technical | 51 | Login issues vs account access |
| Technical → Account | 43 | App crashes affecting login |

### Business Recommendations

1. **Review Account vs Billing boundary:** 170 combined misclassifications. Consider human review for tickets containing both "account" and "payment" keywords.

2. **Refund class (6.6%):** Low volume but high business risk. Route all Refund predictions with confidence < 0.90 to human agents.

3. **Technical Support:** 12.3% of tickets. Consider sub-categorization into "Bug", "Feature Request", "Integration".

---

## 🚀 Future Roadmap

| Version | Feature | Status |
|---------|---------|--------|
| v2.1 | BERT/DistilBERT embeddings | Planned |
| v2.2 | MLflow experiment tracking | Planned |
| v2.3 | FastAPI rate limiting + auth | Planned |
| v2.4 | Prometheus + Grafana monitoring | Planned |
| v2.5 | Real-time drift detection | Planned |
| v3.0 | Multi-label classification | Planned |

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'feat: add amazing feature'`
4. Push to the branch: `git push github feature/amazing-feature`
5. Open a Pull Request

Before submitting:

- ✅ All tests pass
- ✅ Code formatted with Black
- ✅ Lint passes with Ruff
- ✅ Type checks pass with MyPy
- ✅ Documentation updated

---

## 📄 License

This project is licensed under the **MIT License**.

See [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**Mohammad Hussein**

- 🐙 GitHub: [github.com/mohammad-hussein-dev](https://github.com/mohammad-hussein-dev)
- 🦊 GitLab: [gitlab.com/mohammad-hussein-dev](https://gitlab.com/mohammad-hussein-dev)
- 🌐 Portfolio: [mohammad-hussein-dev.github.io](https://mohammad-hussein-dev.github.io)
- 💼 LinkedIn: [mohammad-hussein-dev](https://linkedin.com/in/mohammad-hussein-dev)
- 💬 Telegram: [@mohammad_hussein_dev](https://t.me/mohammad_hussein_dev)
- 📧 Email: [king.mohamd.09876@gmail.com](mailto:king.mohamd.09876@gmail.com)

> *"Clean code, solid architecture, and data-driven decisions."*

---

⭐ **If you find this project useful, please consider giving it a star on GitHub!**

Built with 🐧💻 in Arch Linux

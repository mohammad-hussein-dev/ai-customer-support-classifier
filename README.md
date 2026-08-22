# 🤖 AI Customer Support Ticket Classifier v2.0

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange?style=flat-square)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-ff4b4b?style=flat-square)](https://streamlit.io)
[![Rich](https://img.shields.io/badge/Rich-13.7%2B-green?style=flat-square)](https://github.com/Textualize/rich)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

> **Production-ready NLP pipeline** for automated banking customer support ticket classification.
> Trained on **BANKING77** dataset with 8 fine-grained intents.
> Features dark-theme visualizations, Rich terminal output, and a glassmorphism Streamlit dashboard.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Architecture](#-architecture)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Evaluation](#-evaluation)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Business Implications](#-business-implications)
- [Author](#-author)

---

## 🎯 Overview

Customer support teams in financial services receive thousands of tickets daily. Manual triage causes:

- **Delays:** Average 4-6 hours response time
- **Errors:** 15-20% misclassification rate
- **Cost:** $2.50 per manual ticket routing
- **Scale:** Cannot handle peak loads

This project solves these problems with an end-to-end ML pipeline that:

1. **Classifies** tickets into 8 banking intents automatically
2. **Explains** why each prediction was made (TF-IDF coefficients)
3. **Monitors** confidence and routes low-confidence tickets to humans
4. **Deploys** as an interactive Streamlit dashboard

### Selected Intents (BANKING77 Subset)

| # | Intent | Description |
|---|--------|-------------|
| 1 | `card_arrival` | Card delivery issues |
| 2 | `card_not_working` | Card malfunction |
| 3 | `cash_withdrawal_not_recognised` | Unknown ATM withdrawals |
| 4 | `declined_card_payment` | Payment rejection |
| 5 | `lost_or_stolen_card` | Security incidents |
| 6 | `transaction_charged_twice` | Duplicate charges |
| 7 | `transfer_not_received_by_recipient` | Missing transfers |
| 8 | `cash_withdrawal_charge` | ATM fee disputes |

---

## 📊 Dataset

**BANKING77** — Public English-language benchmark for fine-grained online-banking intent classification.

- **Total:** 13,083 customer-service queries
- **Train:** 10,003 examples
- **Test:** 3,080 examples
- **Intents:** 77 (we use a manageable subset of 8)
- **License:** CC BY 4.0
- **Source:** [PolyAI GitHub](https://github.com/PolyAI-LDN/task-specific-datasets)

---

## 🏗️ Architecture

```
Raw Ticket
    |
    v
TextPreprocessor (NLTK + Banking Stopwords)
    - Lowercase
    - Regex cleaning (URLs, emails, numbers)
    - Tokenization
    - Lemmatization
    - Domain-aware stopword removal
    |
    v
TF-IDF Vectorizer (scikit-learn Pipeline)
    - N-gram: (1, 2)
    - Max features: 5,000
    - Sublinear TF
    - Min DF: 2 | Max DF: 0.95
    |
    v
Logistic Regression (balanced class weights)
    - C = 1.0
    - 8-class output
    - Probability calibration
    |
    v
Intent + Confidence + Explainability + Human Review Flag
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎨 **Dark Theme** | Publication-quality figures with custom dark palette |
| 📊 **Combined Dashboards** | Single figure with 6 subplots for EDA and evaluation |
| 🖥️ **Rich Terminal** | Beautiful color-coded tables and progress output |
| 🌐 **Streamlit App** | Glassmorphism UI with interactive Plotly charts |
| 🔍 **Explainability** | Top TF-IDF terms for every prediction |
| ⚠️ **Human Review** | Low-confidence routing with configurable threshold |
| 📈 **Calibration** | Reliability diagrams for confidence assessment |
| 🏦 **Banking-Optimized** | Domain-specific preprocessing and stopwords |

---

## 🚀 Installation

### Requirements

- Python 3.10+
- Arch Linux / macOS / Ubuntu

### Setup

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

---

## 📖 Usage

### 1. Train & Evaluate

```bash
make train
```

Or manually:

```bash
python scripts/train_and_evaluate.py
```

**Output:**
- `models/production_v2/` — Serialized model, vectorizer, encoder
- `reports/figures/` — Dark-theme evaluation dashboards
- `reports/tables/evaluation_results.json` — Structured metrics

### 2. Launch Streamlit Dashboard

```bash
make streamlit
```

Or:

```bash
streamlit run deployment/app.py
```

**Features:**
- 🎯 Single ticket prediction with confidence gauge
- 📁 Batch CSV upload with distribution charts
- 📊 Model performance metrics
- 🔍 Top contributing terms per prediction

### 3. Error Analysis

```bash
make evaluate
```

### 4. Run Tests

```bash
make test
```

---

## 📁 Project Structure

```
ai-customer-support-classifier/
│
├── config/
│   └── config.yaml              # Centralized configuration
│
├── src/
│   ├── data/
│   │   └── preprocessing.py     # TextPreprocessor (NLTK + banking stopwords)
│   ├── features/
│   │   └── build_features.py    # FeatureBuilder (TF-IDF + metadata)
│   ├── models/
│   │   ├── train_model.py       # ModelTrainer (CV + GridSearch)
│   │   ├── evaluate_model.py    # ModelEvaluator (Rich output + dark figures)
│   │   └── predict_model.py     # TicketClassifier (production inference)
│   ├── utils/
│   │   ├── config.py            # YAML config loader
│   │   ├── helpers.py           # Common utilities
│   │   ├── logger.py            # Rich logging setup
│   │   └── metrics.py           # Beautiful metric tables
│   └── visualization/
│       └── visualize.py         # BankingVisualizer (dark theme dashboards)
│
├── scripts/
│   ├── train_and_evaluate.py    # End-to-end training pipeline
│   └── error_analysis.py        # Deep error analysis with recommendations
│
├── deployment/
│   └── app.py                   # Streamlit dashboard (glassmorphism)
│
├── models/
│   └── production_v2/           # Serialized artifacts
│
├── reports/
│   ├── figures/                 # Dark-theme visualizations
│   └── tables/                  # JSON/CSV evaluation reports
│
├── requirements.txt
├── Makefile
└── README.md
```

---

## 📈 Evaluation

### Required Visualizations (StudyBuild)

| # | Figure | File |
|---|--------|------|
| 1 | **Intent Distribution** | `00_eda_master_dashboard.png` (combined) |
| 2 | **Message/Word Length by Intent** | `00_eda_master_dashboard.png` (combined) |
| 3 | **Top TF-IDF Terms** | `02_tfidf_top_terms.png` |
| 4 | **Confusion Matrix** | `01_evaluation_dashboard.png` (combined) |
| 5 | **Per-Class F1-Score** | `01_evaluation_dashboard.png` (combined) |
| 6 | **Confidence Distribution** | `03_confidence_analysis.png` |
| 7 | **Baseline vs ML** | `05_baseline_comparison.png` |

### Metrics

- **Accuracy:** Not reported alone (per requirements)
- **Macro-F1:** Primary metric for class imbalance
- **Per-Class:** Precision, Recall, F1 for all 8 intents
- **Confidence Calibration:** Reliability diagram included

---

## 🎨 Streamlit Dashboard

### Screenshots

The dashboard features:
- **Dark gradient background** with glassmorphism cards
- **Confidence gauge** with color-coded thresholds
- **Interactive probability bars** (Plotly)
- **Batch upload** with pie chart distribution
- **Model performance** metrics table

### Run

```bash
streamlit run deployment/app.py
```

---

## 💼 Business Implications

### Q6: Are all errors equally costly?

**Business-critical intents** (flagged in config):

| Intent | False Negative Risk | Recommended Action |
|--------|---------------------|-------------------|
| `lost_or_stolen_card` | 🔴 **CRITICAL** | Always route to security team |
| `declined_card_payment` | 🟡 **HIGH** | Priority queue for payment team |
| `transaction_charged_twice` | 🟡 **HIGH** | Fast-track to billing team |

**Metric Priority:**
- **Recall** is most important for critical intents (don't miss them)
- **Precision** matters for high-volume intents (don't waste agent time)
- **Macro-F1** balances both across all classes

### Q8: When should the system defer to a human?

**Rule:** Confidence < 0.7 → Human review

**Justification:**
- Covers ~15-20% of predictions
- Captures most high-cost errors
- Balances automation vs accuracy
- Adjustable per intent (stricter for critical intents)

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **Languages** | Python 3.10+ |
| **ML / NLP** | scikit-learn, NLTK, NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Dashboard** | Streamlit |
| **Terminal UI** | Rich |
| **API** | FastAPI, Uvicorn, Pydantic |
| **OS** | Arch Linux |

---

## 🧪 Scientific Requirements

✅ **No data leakage** — TF-IDF fit on training data only  
✅ **Reproducibility** — Fixed random seed (42)  
✅ **Beyond Accuracy** — Macro-F1 + per-class metrics  
✅ **Error Analysis** — 20+ misclassifications inspected  
✅ **Baseline Comparison** — Keyword system vs ML model  
✅ **Interpretability** — TF-IDF coefficients + top terms  

---

## 👨‍💻 Author

**Mohammad Hussein**

- 🐙 GitHub: [github.com/mohammad-hussein-dev](https://github.com/mohammad-hussein-dev)
- 💼 LinkedIn: [mohammad-hussein-dev](https://linkedin.com/in/mohammad-hussein-dev)
- 💬 Telegram: [@mohammad_hussein_dev](https://t.me/mohammad_hussein_dev)
- 📧 Email: [king.mohamd.09876@gmail.com](mailto:king.mohamd.09876@gmail.com)

> *"Clean code, solid architecture, and data-driven decisions."*

---

⭐ **If you find this project useful, please consider giving it a star!**

Built with 🐧💻 in Arch Linux

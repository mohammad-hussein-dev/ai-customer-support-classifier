# 🤖 AI Customer Support Ticket Classifier v2.0

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange?style=flat-square)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-ff4b4b?style=flat-square)](https://streamlit.io)
[![Rich](https://img.shields.io/badge/Rich-13.7%2B-green?style=flat-square)](https://github.com/Textualize/rich)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

> **Production-ready NLP pipeline** for automated banking customer support ticket classification.
> Trained on the **BANKING77** dataset using a focused subset of 8 fine-grained banking intents.
> Includes reproducible preprocessing, feature engineering, model training, evaluation, error analysis, testing, and a Streamlit deployment interface.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Architecture](#-architecture)
- [Key Features](#-key-features)
- [Project Report](#-project-report)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Evaluation](#-evaluation)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Business Implications](#-business-implications)
- [Scientific Requirements](#-scientific-requirements)
- [Technology Stack](#-technology-stack)
- [Author](#-author)

---

## 🎯 Overview

Customer support teams in financial services receive large volumes of customer tickets.
Manual triage can introduce delays, inconsistent routing, and unnecessary operational costs.

This project implements an end-to-end machine-learning pipeline for automatically classifying banking customer-support tickets into **8 selected intents** from the BANKING77 dataset.

The system provides:

1. **Automatic ticket classification**
2. **Confidence estimation**
3. **Prediction explainability using TF-IDF features**
4. **Low-confidence human-review routing**
5. **Model evaluation and error analysis**
6. **Reproducible training and preprocessing**
7. **Interactive Streamlit deployment**
8. **Automated tests for core components and the full pipeline**

### Selected Intents

| # | Intent | Description |
|---|--------|-------------|
| 1 | `card_arrival` | Card delivery issues |
| 2 | `card_not_working` | Card malfunction |
| 3 | `cash_withdrawal_not_recognised` | Unknown ATM withdrawals |
| 4 | `declined_card_payment` | Payment rejection |
| 5 | `lost_or_stolen_card` | Lost or stolen card incidents |
| 6 | `transaction_charged_twice` | Duplicate transaction charges |
| 7 | `transfer_not_received_by_recipient` | Missing transfers |
| 8 | `cash_withdrawal_charge` | ATM fee disputes |

---

## 📊 Dataset

**BANKING77** is a public English-language benchmark dataset for fine-grained online-banking intent classification.

| Property | Value |
|----------|-------|
| Total examples | 13,083 |
| Training examples | 10,003 |
| Test examples | 3,080 |
| Total intents | 77 |
| Intents used | 8 |
| License | CC BY 4.0 |

Source:

[PolyAI Task-Specific Datasets](https://github.com/PolyAI-LDN/task-specific-datasets)

---

## 🏗️ Architecture

```text
                         Raw Customer Ticket
                                  |
                                  v
                    +-------------------------+
                    |    Text Preprocessing   |
                    |-------------------------|
                    | Lowercasing             |
                    | Regex cleaning          |
                    | Tokenization             |
                    | Lemmatization            |
                    | Domain stopword removal |
                    +------------+------------+
                                 |
                                 v
                    +-------------------------+
                    |     Feature Building    |
                    |-------------------------|
                    | TF-IDF                  |
                    | Unigrams + Bigrams      |
                    | Metadata features       |
                    +------------+------------+
                                 |
                                 v
                    +-------------------------+
                    |     ML Classification   |
                    |-------------------------|
                    | Logistic Regression     |
                    | Balanced class weights  |
                    | 8-class classification  |
                    +------------+------------+
                                 |
                                 v
                    +-------------------------+
                    |      Prediction Layer   |
                    |-------------------------|
                    | Intent                  |
                    | Confidence              |
                    | Explainability          |
                    | Human-review flag       |
                    +------------+------------+
                                 |
                    +------------+-------------+
                    |                          |
              High confidence            Low confidence
                    |                          |
                    v                          v
             Automated route             Human review
````

---

## ✨ Key Features

| Feature                    | Description                                                          |
| -------------------------- | -------------------------------------------------------------------- |
| 🧹 **Text Preprocessing**  | NLTK-based preprocessing with domain-aware banking stopwords         |
| 🔢 **TF-IDF Features**     | Unigram and bigram text representation                               |
| 🤖 **ML Classification**   | Logistic Regression for 8-class intent classification                |
| 🔍 **Explainability**      | TF-IDF coefficients and influential terms                            |
| ⚠️ **Human Review**        | Low-confidence predictions can be routed to human agents             |
| 📊 **Evaluation**          | Macro-F1, precision, recall, confusion matrix, and per-class metrics |
| 🔬 **Error Analysis**      | Inspection of model failures and actionable recommendations          |
| 🧪 **Testing**             | Unit and integration tests                                           |
| 🌐 **Streamlit Dashboard** | Interactive prediction and model-analysis interface                  |
| 📈 **Visualization**       | EDA, evaluation, confidence, and model-comparison figures            |
| ⚙️ **Configuration**       | Centralized YAML-based configuration                                 |
| 📝 **Reporting**           | Complete project report available as a PDF                           |

---

# 📄 Project Report

A complete technical report documenting the project's methodology, data preparation, exploratory analysis, model development, evaluation, error analysis, and conclusions is included in the repository.

### 📘 Full Technical Report

**[📄 Open / Download the Project Report](reports/report.pdf)**

The report covers:

* Dataset and problem formulation
* Exploratory Data Analysis (EDA)
* Text preprocessing
* Feature engineering
* Model development
* Baseline comparison
* Model evaluation
* Confusion-matrix analysis
* Per-class performance
* Confidence analysis
* Error analysis
* Business implications
* Conclusions and recommendations

> **The PDF is stored directly in the repository at `reports/report.pdf`.**
> Click the link above to open the full report in GitHub or GitLab and download it as a PDF.

---

## 🚀 Installation

### Requirements

* Python 3.10+
* Linux, macOS, or Windows
* pip
* Git

### Setup

```bash
# Clone repository
git clone https://github.com/mohammad-hussein-dev/ai-customer-support-classifier.git

# Enter project directory
cd ai-customer-support-classifier

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK resources
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## 📖 Usage

### 1. Build the Dataset

```bash
python scripts/build_hybrid_dataset.py
```

### 2. Preprocess and Split Data

```bash
python scripts/preprocess_and_split.py
```

### 3. Train and Evaluate

```bash
python scripts/train_and_evaluate.py
```

The pipeline generates model artifacts and evaluation outputs under:

```text
models/
reports/figures/
reports/tables/
```

### 4. Error Analysis

```bash
python scripts/error_analysis.py
```

### 5. Run Tests

```bash
pytest
```

Or using the project Makefile:

```bash
make test
```

### 6. Launch the Streamlit Dashboard

```bash
streamlit run deployment/app.py
```

---

## 📁 Project Structure

```text
ai-customer-support-classifier/
│
├── api/
│   └── main.py
│
├── config/
│   ├── config.yaml
│   ├── logging.yaml
│   └── model_params.yaml
│
├── data/
│   ├── external/
│   ├── processed/
│   │   ├── test.csv
│   │   ├── train.csv
│   │   ├── y_test.csv
│   │   ├── y_test.npy
│   │   ├── y_train.csv
│   │   └── y_train.npy
│   └── raw/
│       ├── hybrid_dataset.csv
│       └── tickets.csv
│
├── deployment/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── models/
│   ├── baseline/
│   │   ├── feature_builder.pkl
│   │   ├── logistic_regression.pkl
│   │   ├── naive_bayes.pkl
│   │   ├── preprocessor.pkl
│   │   └── svm.pkl
│   │
│   ├── production/
│   │   ├── feature_builder.pkl
│   │   ├── model.pkl
│   │   └── preprocessor.pkl
│   │
│   └── production_v2/
│       ├── encoder.pkl
│       ├── metrics.json
│       ├── model.pkl
│       └── vectorizer.pkl
│
├── notebooks/
│
├── reports/
│   ├── developer.jpg
│   ├── figures/
│   │   ├── 01_intent_distribution.png
│   │   ├── 02_text_length_violin.png
│   │   ├── 03_confusion_matrix.png
│   │   └── 04_per_class_f1.png
│   ├── report.pdf
│   ├── report.tex
│   └── tables/
│       ├── eda_summary.csv
│       ├── error_analysis.json
│       ├── evaluation_report.txt
│       └── model_comparison.csv
│
├── scripts/
│   ├── build_hybrid_dataset.py
│   ├── error_analysis.py
│   ├── preprocess_and_split.py
│   ├── train_and_evaluate.py
│   └── visualize.py
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── make_dataset.py
│   │   └── preprocessing.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build_features.py
│   │   └── text_features.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── evaluate_model.py
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── helpers.py
│   │   ├── logger.py
│   │   └── metrics.py
│   │
│   └── visualization/
│       ├── __init__.py
│       └── visualize.py
│
├── tests/
│   ├── __init__.py
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_pipeline.py
│   └── unit/
│       ├── __init__.py
│       ├── test_features.py
│       ├── test_model.py
│       └── test_preprocessing.py
│
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── pyproject.toml
├── README.md
└── requirements.txt
```

---

## 📈 Evaluation

The project evaluates the classifier beyond simple accuracy.

### Evaluation Components

| Component               | Purpose                                        |
| ----------------------- | ---------------------------------------------- |
| **Macro-F1**            | Primary overall metric across the 8 classes    |
| **Precision**           | Measures correctness of positive predictions   |
| **Recall**              | Measures coverage of actual intent classes     |
| **Per-Class F1**        | Identifies weak and strong intent classes      |
| **Confusion Matrix**    | Reveals systematic class confusion             |
| **Confidence Analysis** | Evaluates prediction certainty                 |
| **Error Analysis**      | Investigates representative misclassifications |
| **Baseline Comparison** | Compares simple baseline approaches against ML |

### Generated Reports

Evaluation artifacts are stored under:

```text
reports/
├── figures/
└── tables/
```

The complete interpretation of the results is available in:

**[📄 Project Report](reports/report.pdf)**

---

## 🎨 Streamlit Dashboard

The project includes an interactive Streamlit interface for model inference and analysis.

### Features

* 🎯 Single-ticket prediction
* 📊 Prediction probabilities
* 🔍 Top contributing terms
* ⚠️ Confidence-based human-review flag
* 📁 Batch CSV prediction
* 📈 Prediction distribution
* 📋 Model performance information

### Run

```bash
streamlit run deployment/app.py
```

---

## 💼 Business Implications

### Are all errors equally costly?

No.

Certain banking intents have substantially higher operational or security consequences.

| Intent                      | Risk            | Recommended Action              |
| --------------------------- | --------------- | ------------------------------- |
| `lost_or_stolen_card`       | 🔴 **Critical** | Immediate security-team routing |
| `declined_card_payment`     | 🟡 **High**     | Priority payment-team routing   |
| `transaction_charged_twice` | 🟡 **High**     | Fast-track to billing support   |

### Metric Priorities

* **Recall** is particularly important for critical intents.
* **Precision** matters for high-volume operational intents.
* **Macro-F1** provides balanced evaluation across all selected classes.

---

## 👤 Human Review Policy

The classifier can defer uncertain predictions to human agents.

### Default Rule

```text
confidence < 0.70
        ↓
Human Review
```

This threshold can be adjusted according to operational requirements.

The objective is not to automate every decision, but to combine:

```text
High-confidence predictions
          +
Human review for uncertain cases
          =
Safer automation
```

---

## 🧪 Scientific Requirements

The project follows several reproducibility and machine-learning best practices:

* ✅ **No data leakage** — preprocessing and feature fitting are performed using training data appropriately.
* ✅ **Reproducibility** — fixed random seed where applicable.
* ✅ **Beyond Accuracy** — Macro-F1 and per-class metrics are included.
* ✅ **Error Analysis** — representative model errors are investigated.
* ✅ **Baseline Comparison** — baseline approaches are compared against the ML classifier.
* ✅ **Interpretability** — TF-IDF features and model coefficients provide prediction-level insight.
* ✅ **Automated Testing** — unit and integration tests cover core functionality.
* ✅ **Configuration Management** — model and application parameters are centralized in configuration files.

---

## 🛠️ Technology Stack

| Category             | Technologies                      |
| -------------------- | --------------------------------- |
| **Language**         | Python 3.10+                      |
| **ML / NLP**         | scikit-learn, NLTK, NumPy, Pandas |
| **Visualization**    | Matplotlib, Seaborn, Plotly       |
| **Dashboard**        | Streamlit                         |
| **Terminal UI**      | Rich                              |
| **API**              | FastAPI, Uvicorn, Pydantic        |
| **Testing**          | Pytest                            |
| **Containerization** | Docker, Docker Compose            |
| **Configuration**    | YAML                              |
| **OS**               | Linux / Arch Linux                |

---

## 👨‍💻 Author

**Mohammad Hussein**

* 🐙 GitHub: [@mohammad-hussein-dev](https://github.com/mohammad-hussein-dev)
* 💼 LinkedIn: [mohammad-hussein-dev](https://linkedin.com/in/mohammad-hussein-dev)
* 💬 Telegram: [@mohammad_hussein_dev](https://t.me/mohammad_hussein_dev)
* 📧 Email: [king.mohamd.09876@gmail.com](mailto:king.mohamd.09876@gmail.com)

> *"Clean code, solid architecture, and data-driven decisions."*

---

⭐ **If you find this project useful, please consider giving it a star!**

Built with 🐧💻 on Arch Linux.

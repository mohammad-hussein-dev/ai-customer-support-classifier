#!/usr/bin/env python3
"""Generate all remaining project files."""

import os

PROJECT_ROOT = os.path.expanduser("~/Projects/ai-customer-support-classifier")
os.chdir(PROJECT_ROOT)

files = {}

files["README.md"] = """# AI Customer Support Ticket Classifier\n\n[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)\n[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)\n[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)\n\n> An end-to-end machine learning pipeline for automated classification of customer support tickets using TF-IDF and Logistic Regression.\n\n## Features\n\n- **4 Categories**: Billing, Technical Support, Account, Refund\n- **5,000 Synthetic Tickets** with realistic class imbalance\n- **Complete ML Pipeline**: Preprocessing -> Feature Engineering -> Training -> Evaluation -> Deployment\n- **Production-Ready**: Confidence thresholding with human-in-the-loop routing\n- **Interactive App**: Streamlit web interface for real-time classification\n- **Comprehensive EDA**: 20+ publication-quality visualizations\n- **Cross-Validation**: Stratified 5-fold CV for reliable metrics\n\n## Quick Start\n\n```bash\n# Clone and setup\ngit clone <repo-url>\ncd ai-customer-support-classifier\npython3 -m venv venv --prompt=ticket-classifier\nsource venv/bin/activate\npip install -r requirements.txt\n\n# Run full pipeline\npython notebooks/exploratory/train_and_evaluate.py\n\n# Launch Streamlit app\nstreamlit run deployment/app.py\n```\n\n## Project Structure\n\n```\nai-customer-support-classifier/\n├── config/                 # YAML configuration files\n├── data/\n│   ├── raw/               # Original datasets\n│   └── processed/         # Cleaned datasets\n├── deployment/\n│   └── app.py             # Streamlit application\n├── models/\n│   ├── baseline/          # Experimental models\n│   └── production/        # Deployed artifacts\n├── notebooks/\n│   └── exploratory/       # EDA and experiments\n├── reports/\n│   ├── figures/           # Visualizations\n│   └── report.pdf         # LaTeX report\n├── src/\n│   ├── data/              # Preprocessing\n│   ├── features/          # Feature engineering\n│   ├── models/            # Training & evaluation\n│   └── visualization/     # Plotting utilities\n└── tests/                 # Unit & integration tests\n```\n\n## Model Performance\n\n| Model | Accuracy | F1 (Weighted) | F1 (Macro) |\n|-------|----------|---------------|------------|\n| Logistic Regression | 1.0000 | 1.0000 | 1.0000 |\n| Linear SVM | 1.0000 | 1.0000 | 1.0000 |\n| Naive Bayes | 1.0000 | 1.0000 | 1.0000 |\n\n*Note: Perfect scores reflect the synthetic nature of the dataset. Real-world deployment would use more diverse data.*\n\n## API Usage\n\n```python\nfrom src.models.predict_model import TicketClassifier\n\nclf = TicketClassifier.from_directory("models/production")\nresult = clf.predict("I was charged twice this month.")\n\nprint(result["category"])      # Billing\nprint(result["confidence"])    # 0.9330\nprint(result["needs_review"])  # False\n```\n\n## Configuration\n\nAll hyperparameters are centralized in `config/config.yaml`:\n\n```yaml\nfeatures:\n  vectorizer: "tfidf"\n  max_features: 10000\n  ngram_range: [1, 2]\n\nmodel:\n  name: "logistic_regression"\n  class_weight: "balanced"\n  C: 1.0\n```\n\n## Docker\n\n```bash\ndocker-compose up --build\n# App available at http://localhost:8501\n```\n\n## License\n\nMIT License - see [LICENSE](LICENSE) file."""

files["Dockerfile"] = """FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"

EXPOSE 8501

CMD ["streamlit", "run", "deployment/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""

files["docker-compose.yml"] = """version: "3.8"

services:
  classifier:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - PYTHONUNBUFFERED=1
"""

files[".github/workflows/ci.yml"] = """name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt

      - name: Lint with flake8
        run: flake8 src tests --max-line-length=88

      - name: Type check with mypy
        run: mypy src

      - name: Test with pytest
        run: pytest tests/ -v --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
"""

files[".github/workflows/cd.yml"] = """name: CD

on:
  push:
    tags:
      - "v*"

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t ticket-classifier:${{ github.ref_name }} .

      - name: Test Docker image
        run: |
          docker run -d -p 8501:8501 --name test ticket-classifier:${{ github.ref_name }}
          sleep 10
          curl -f http://localhost:8501 || exit 1
          docker stop test
"""

files["tests/unit/test_preprocessing.py"] = '''"""Unit tests for text preprocessing."""

import pytest
from src.data.preprocessing import TextPreprocessor


class TestTextPreprocessor:
    """Test suite for TextPreprocessor."""

    def test_default_initialization(self):
        preprocessor = TextPreprocessor()
        assert preprocessor.lowercase is True
        assert preprocessor.remove_punctuation is True

    def test_clean_lowercase(self):
        preprocessor = TextPreprocessor()
        result = preprocessor.clean("HELLO WORLD")
        assert result == "hello world"

    def test_clean_punctuation_removal(self):
        preprocessor = TextPreprocessor()
        result = preprocessor.clean("Hello, world!!!")
        assert "," not in result
        assert "!" not in result

    def test_clean_stopword_removal(self):
        preprocessor = TextPreprocessor()
        result = preprocessor.clean("this is a test")
        assert "this" not in result.split()
        assert "is" not in result.split()

    def test_transform_list(self):
        preprocessor = TextPreprocessor()
        texts = ["Hello world", "Test sentence"]
        results = preprocessor.transform(texts)
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    def test_non_string_input(self):
        preprocessor = TextPreprocessor()
        result = preprocessor.clean(12345)
        assert result == ""
'''

files["tests/unit/test_features.py"] = '''"""Unit tests for feature engineering."""

import numpy as np
from scipy.sparse import csr_matrix

from src.features.build_features import FeatureBuilder


class TestFeatureBuilder:
    """Test suite for FeatureBuilder."""

    def test_initialization(self):
        builder = FeatureBuilder()
        assert builder.max_features == 10000
        assert builder.ngram_range == (1, 2)

    def test_fit_transform(self):
        texts = [
            "hello world test",
            "hello python code",
            "test machine learning",
        ]
        builder = FeatureBuilder(max_features=100)
        matrix = builder.fit_transform(texts)
        assert isinstance(matrix, csr_matrix)
        assert matrix.shape[0] == 3
        assert matrix.shape[1] <= 100

    def test_vocabulary_size(self):
        texts = ["hello world", "hello test"]
        builder = FeatureBuilder(max_features=10)
        builder.fit(texts)
        assert len(builder.get_feature_names()) > 0

    def test_transform_new_text(self):
        texts = ["hello world", "python code"]
        builder = FeatureBuilder(max_features=10)
        builder.fit(texts)
        new_matrix = builder.transform(["hello python"])
        assert new_matrix.shape[0] == 1
'''

files["tests/unit/test_model.py"] = '''"""Unit tests for model training."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.models.train_model import ModelTrainer


class TestModelTrainer:
    """Test suite for ModelTrainer."""

    def test_initialization(self):
        trainer = ModelTrainer(model_name="logistic_regression")
        assert trainer.model_name == "logistic_regression"

    def test_unsupported_model(self):
        with pytest.raises(ValueError):
            ModelTrainer(model_name="random_forest")

    def test_cross_validate(self):
        np.random.seed(42)
        X = csr_matrix(np.random.rand(100, 10))
        y = np.random.choice(["A", "B", "C"], size=100)

        trainer = ModelTrainer(model_name="logistic_regression", cv_folds=3)
        results = trainer.cross_validate(X, y)

        assert "test_accuracy" in results
        assert "test_f1_weighted" in results
        assert len(results["test_accuracy"]) == 3

    def test_fit_predict(self):
        np.random.seed(42)
        X = csr_matrix(np.random.rand(50, 10))
        y = np.random.choice(["A", "B"], size=50)

        trainer = ModelTrainer(model_name="naive_bayes")
        trainer.fit(X, y)

        X_test = csr_matrix(np.random.rand(5, 10))
        predictions = trainer.predict(X_test)

        assert len(predictions) == 5
        assert all(isinstance(p, str) for p in predictions)
'''

files["tests/integration/test_pipeline.py"] = '''"""Integration tests for end-to-end pipeline."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.data.preprocessing import TextPreprocessor
from src.features.build_features import FeatureBuilder
from src.models.predict_model import TicketClassifier
from src.models.train_model import ModelTrainer


class TestPipeline:
    """End-to-end pipeline integration tests."""

    def test_full_pipeline(self, tmp_path):
        df = pd.DataFrame({
            "text": [
                "I was charged twice for my subscription",
                "The app crashes when I upload files",
                "I forgot my password and cannot login",
                "I want a refund for my purchase",
            ] * 5,
            "category": ["Billing", "Technical Support", "Account", "Refund"] * 5,
        })

        preprocessor = TextPreprocessor()
        texts_clean = preprocessor.transform(df["text"].tolist())

        builder = FeatureBuilder(max_features=50)
        X = builder.fit_transform(texts_clean)
        y = np.array(df["category"].tolist())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        trainer = ModelTrainer(model_name="logistic_regression")
        trainer.fit(X_train, y_train)

        y_pred = trainer.predict(X_test)
        assert len(y_pred) == len(y_test)

        model_path = tmp_path / "model.pkl"
        prep_path = tmp_path / "preprocessor.pkl"
        feat_path = tmp_path / "feature_builder.pkl"

        joblib.dump(trainer.model, model_path)
        joblib.dump(preprocessor, prep_path)
        joblib.dump(builder, feat_path)

        clf = TicketClassifier(
            model_path=str(model_path),
            preprocessor_path=str(prep_path),
            vectorizer_path=str(feat_path),
        )
        result = clf.predict("I was charged twice")
        assert "category" in result
        assert "confidence" in result
        assert "needs_review" in result
'''

for path, content in files.items():
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"✅ {path}")

print("\n🎉 All files generated successfully!")
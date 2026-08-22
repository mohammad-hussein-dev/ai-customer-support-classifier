# Banking Intent Classifier — Makefile
# ====================================

.PHONY: help install data train evaluate streamlit test clean

PYTHON := python3
PIP := pip3

help:
	@echo "🏦 Banking Intent Classifier — Available Commands:"
	@echo ""
	@echo "  make install      Install dependencies"
	@echo "  make data         Build and preprocess dataset"
	@echo "  make train        Train model and generate reports"
	@echo "  make evaluate     Run error analysis"
	@echo "  make streamlit    Launch Streamlit dashboard"
	@echo "  make api          Launch FastAPI server"
	@echo "  make test         Run test suite"
	@echo "  make clean        Remove generated artifacts"
	@echo ""

install:
	$(PIP) install -r requirements.txt
	$(PYTHON) -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

data:
	@echo "📊 Building dataset..."
	$(PYTHON) scripts/build_hybrid_dataset.py || true
	$(PYTHON) scripts/preprocess_and_split.py || true

train:
	@echo "🚀 Training pipeline..."
	$(PYTHON) scripts/train_and_evaluate.py

evaluate:
	@echo "🔍 Running error analysis..."
	$(PYTHON) scripts/error_analysis.py

streamlit:
	@echo "🎨 Launching Streamlit dashboard..."
	streamlit run deployment/app.py

api:
	@echo "🌐 Launching FastAPI server..."
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	@echo "🧪 Running tests..."
	pytest tests/ -v --tb=short

clean:
	@echo "🧹 Cleaning artifacts..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	rm -rf reports/figures/*.png reports/tables/*.json 2>/dev/null || true
	@echo "✅ Clean complete"

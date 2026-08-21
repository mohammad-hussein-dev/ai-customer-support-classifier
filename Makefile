.PHONY: help install test lint format clean docker-build docker-run deploy report

PYTHON := python3
PIP := pip3

help:
	@echo "Available commands:"
	@echo "  install      Install dependencies"
	@echo "  test         Run test suite"
	@echo "  lint         Run code quality checks"
	@echo "  format       Auto-format code with black"
	@echo "  clean        Remove generated artifacts"
	@echo "  report       Generate LaTeX PDF report"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run Docker container"
	@echo "  deploy       Start Streamlit app"

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	flake8 src tests
	mypy src
	pylint src

format:
	black src tests
	isort src tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache htmlcov
	rm -rf reports/*.pdf reports/figures/*.png

report:
	cd reports && pdflatex report.tex && pdflatex report.tex
	@echo "✅ Report generated: reports/report.pdf"

docker-build:
	docker build -t ticket-classifier:latest .

docker-run:
	docker run -p 8501:8501 ticket-classifier:latest

deploy:
	streamlit run deployment/app.py

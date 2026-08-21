"""Setup script for AI Customer Support Ticket Classifier."""

from setuptools import find_packages, setup

setup(
    name="ai-customer-support-classifier",
    version="1.0.0",
    description="AI-powered customer support ticket classification system",
    author="Mohammad-Hussein",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "nltk>=3.8.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "joblib>=1.3.0",
        "pyyaml>=6.0",
    ],
)

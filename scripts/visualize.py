#!/usr/bin/env python3
"""Generate required visualizations for the project report."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import json

# Load data
df = pd.read_csv('data/raw/hybrid_dataset.csv')
test_df = pd.read_csv('data/processed/test.csv')
model = joblib.load('models/production_v2/model.pkl')
vectorizer = joblib.load('models/production_v2/vectorizer.pkl')
encoder = joblib.load('models/production_v2/encoder.pkl')

# 1. Intent distribution
plt.figure(figsize=(10,6))
counts = df['category'].value_counts()
counts.plot(kind='bar', color='skyblue')
plt.title('Intent Distribution')
plt.xlabel('Intent')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('reports/figures/01_intent_distribution.png', dpi=150)
plt.close()

# 2. Text length distribution by intent
df['text_length'] = df['text'].str.len()
plt.figure(figsize=(12,6))
sns.violinplot(x='category', y='text_length', data=df)
plt.title('Text Length Distribution by Intent')
plt.xlabel('Intent')
plt.ylabel('Text Length (characters)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('reports/figures/02_text_length_violin.png', dpi=150)
plt.close()

# 3. Confusion matrix
y_pred = model.predict(vectorizer.transform(test_df['text_clean']))
cm = confusion_matrix(test_df['category'], encoder.inverse_transform(y_pred))
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('reports/figures/03_confusion_matrix.png', dpi=150)
plt.close()

# 4. Per-class F1 scores (try to load from metrics.json)
metrics_path = Path('models/production_v2/metrics.json')
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)
    best_model = metrics['best_model']
    per_class_f1 = metrics['models'][best_model]['per_class_f1']
    classes = list(per_class_f1.keys())
    f1_scores = list(per_class_f1.values())
    plt.figure(figsize=(10,6))
    plt.bar(classes, f1_scores, color='orange')
    plt.title(f'Per-Class F1 Scores ({best_model})')
    plt.xlabel('Intent')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('reports/figures/04_per_class_f1.png', dpi=150)
    plt.close()
else:
    print("⚠️  metrics.json not found. Skipping F1 chart.")

# 5. Top TF-IDF terms (for Logistic Regression if available)
if hasattr(model, 'coef_'):
    feature_names = vectorizer.get_feature_names_out()
    for i, cls in enumerate(encoder.classes_):
        coef = model.coef_[i]
        top_idx = np.argsort(coef)[-10:][::-1]
        top_terms = [feature_names[j] for j in top_idx]
        top_scores = [coef[j] for j in top_idx]
        plt.figure(figsize=(10,6))
        plt.barh(top_terms, top_scores, color='teal')
        plt.title(f'Top 10 TF-IDF Terms for {cls}')
        plt.xlabel('Coefficient')
        plt.tight_layout()
        plt.savefig(f'reports/figures/05_top_terms_{cls}.png', dpi=150)
        plt.close()

print("✅ All visualizations saved to reports/figures/")

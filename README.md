# YouTube Comment Analyzer: End-to-End Machine Learning Experimentation Framework

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-ffe873.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)
![Optuna](https://img.shields.io/badge/Optuna-HPT-blue)

## Abstract

This repository presents a comprehensive machine learning experimentation framework for multi-class YouTube comment classification. The objective is to automatically categorize user-generated comments through a systematic sequence of text preprocessing, feature engineering, imbalance handling, hyperparameter optimization, and model evaluation.

Unlike deployment-focused repositories, this project emphasizes the research and experimentation lifecycle of machine learning systems. Multiple feature representations and model families were evaluated, including sparse text representations (Bag-of-Words and TF-IDF), classical machine learning algorithms, gradient boosting frameworks, and transformer-based architectures.

The project follows a structured progression from baseline modeling to advanced transformer fine-tuning while maintaining experiment reproducibility through MLflow tracking and Optuna-based hyperparameter optimization.

---

# Project Objective

The goal of this project is to develop a robust text classification system capable of automatically classifying YouTube comments into predefined categories.

The repository investigates:

* The impact of text preprocessing on model performance.
* The effectiveness of sparse feature representations.
* The influence of class imbalance handling.
* The benefits of hyperparameter optimization.
* The performance gap between classical machine learning and transformer-based architectures.

---

# Technology Stack

## Core Libraries

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn

## Machine Learning

* Scikit-Learn
* XGBoost
* LightGBM
* CatBoost

## Deep Learning

* PyTorch
* HuggingFace Transformers
* RoBERTa Base

## Optimization & Experiment Tracking

* Optuna
* MLflow

## Imbalance Handling

* class_weights

---

# Repository Structure

```text
.
├── yt_comment_analyzer_preprocessing.ipynb
├── experiment_1_baseline_model.ipynb
├── experiment_2_bow_tfidf.ipynb
├── experiment_3_tfidf_(1,3)_max_features.ipynb
├── experiment_4_handling_imbalanced_data.ipynb
├── custom_features.ipynb
│
├── experiment_5_lor_with_hpt.ipynb
├── experiment_5_naive_bayes_with_hpt.ipynb
├── experiment_5_knn_with_hpt.ipynb
├── experiment_5_svm_with_hpt.ipynb
├── experiment_5_linearsvm_with_hpt.ipynb
├── experiment_5_random_forest_with_hpt.ipynb
├── experiment_5_xgboost_with_hpt.ipynb
├── experiment_5_lightgbm_with_hpt.ipynb
├── experiment_5_catboost_with_hpt.ipynb
├── experiment_5_roberta_base_with_optuna_hpt.ipynb
│
├── mlruns/
└── requirements.txt
```

---

# Experimental Pipeline

## Phase 1: Text Preprocessing

Notebook:

```text
yt_comment_analyzer_preprocessing.ipynb
```

### Objectives

The raw YouTube comments contain substantial noise that negatively impacts downstream classification models.

### Processing Steps

* Missing value handling
* Lowercase normalization
* URL removal
* HTML artifact removal
* Punctuation removal
* Special character filtering
* Whitespace normalization
* Dataset validation

### Outcome

A standardized corpus suitable for machine learning experimentation.

---

## Phase 2: Baseline Model Development

Notebook:

```text
experiment_1_baseline_model.ipynb
```

### Purpose

Establish an initial benchmark before advanced feature engineering and optimization.

### Results

| Metric      | Score  |
| ----------- | ------ |
| Accuracy    | 64.79% |
| Macro F1    | 49.00% |
| Weighted F1 | 57.00% |

### Observation

The baseline model demonstrated limited ability to capture minority classes, highlighting the necessity for improved feature engineering and model optimization.

---

## Phase 3: Sparse Feature Engineering

Notebook:

```text
experiment_2_bow_tfidf.ipynb
```

### Bag-of-Words

Represents text through token frequency counts.

Advantages:

* Simple
* Interpretable
* Fast

Limitations:

* No semantic understanding
* Ignores context

### TF-IDF

Represents text by weighting terms according to their importance within the corpus.

Advantages:

* Reduces impact of common words
* Improves class discrimination
* Generally superior to raw count vectors

### Best Result

| Model                     | Accuracy | Weighted F1 |
| ------------------------- | -------: | ----------: |
| BoW + Logistic Regression |   65.69% |      58.77% |

---

## Phase 4: Advanced TF-IDF Engineering

Notebook:

```text
experiment_3_tfidf_(1,3)_max_features.ipynb
```

### Objective

Capture richer contextual information through n-gram modeling.

### Techniques

* Unigrams
* Bigrams
* Trigrams
* Vocabulary pruning
* Feature dimensionality optimization

### Best Result

| Metric      | Score  |
| ----------- | ------ |
| Accuracy    | 86.24% |
| Macro F1    | 85.07% |
| Weighted F1 | 86.14% |

### Impact

Compared to the baseline model:

* Accuracy improved from 64.79% to 86.24%.
* Macro F1 improved from 49.00% to 85.07%.

This phase contributed the largest performance gain across the entire experimentation pipeline.

---
## Phase 5: Class Imbalance Handling

Notebook: experiment_4_handling_imbalanced_data.ipynb

### Problem

The dataset exhibited class imbalance, causing models to favor majority classes and reducing classification performance on underrepresented categories.

### Approaches Evaluated

Multiple imbalance handling strategies were systematically evaluated:

- Random Oversampling
- Random Undersampling
- SMOTE (Synthetic Minority Oversampling Technique)
- Class-Weighted Learning

### Best Performing Approach

Experimental results showed that **Class-Weighted Learning** outperformed all resampling-based techniques, including SMOTE.

Rather than generating synthetic observations, class weighting adjusts the training objective by assigning higher penalties to misclassified minority-class samples while preserving the original data distribution.

### Best Result

| Model | Imbalance Method | Accuracy | Macro F1 | Weighted F1 |
|---------|---------|---------:|---------:|---------:|
| LinearSVM | Class Weights | **85.41%** | **84.02%** | **85.28%** |

### Observation

Class-weighted optimization consistently achieved better balanced performance than SMOTE, oversampling, and undersampling approaches. The results suggest that preserving the original text distribution while compensating for class imbalance through weighted penalties leads to superior generalization.

### Conclusion

For this dataset, class-weighted learning proved to be the most effective imbalance mitigation strategy. Consequently, later experiments prioritized weighted optimization over synthetic resampling techniques.
---

## Phase 6: Custom Feature Engineering

Notebook:

```text
custom_features.ipynb
```

### Objective

Augment sparse representations with handcrafted linguistic features.

### Examples

* Comment length
* Word count
* Character count
* Lexical diversity
* Structural indicators

### Motivation

These features provide complementary information that is not directly encoded within TF-IDF vectors.

---

# Hyperparameter Optimization

All optimization experiments were conducted using Optuna.

The search process optimized:

* Regularization strength
* Learning rates
* Tree depth
* Number of estimators
* Kernel parameters
* Nearest-neighbor configurations
* Transformer learning rates
* Batch sizes

The optimization objective primarily focused on maximizing Macro F1 Score.

---

# Model Comparison

## Final Experimental Results

| Model                        |   Accuracy |   Macro F1 | Weighted F1 |
| ---------------------------- | ---------: | ---------: | ----------: |
| Baseline Model               |     0.6479 |     0.4900 |      0.5700 |
| TF-IDF + Logistic Regression |     0.6532 |     0.5056 |      0.5787 |
| Naive Bayes (HPT)            |     0.6604 |     0.6447 |      0.6672 |
| KNN (HPT)                    |     0.5268 |     0.4937 |      0.5100 |
| XGBoost (HPT)                |     0.8336 |     0.8203 |      0.8320 |
| LightGBM (HPT)               |     0.8308 |     0.8186 |      0.8291 |
| SVM (HPT)                    |     0.8546 |     0.8427 |      0.8543 |
| Linear SVM (HPT)             |     0.8559 |     0.8418 |      0.8543 |
| Logistic Regression (HPT)    |     0.8715 |     0.8620 |      0.8710 |
| TF-IDF (1,3) Optimized       |     0.8624 |     0.8507 |      0.8614 |
| RoBERTa Base + Optuna        | **0.8887** | **0.8813** |  **0.8893** |

---

# Algorithmic Taxonomy & Rationale

## Linear Models

### Logistic Regression

Selected because:

* Performs exceptionally well on sparse TF-IDF features.
* Highly interpretable.
* Computationally efficient.

Final Performance:

* Accuracy: 87.15%
* Macro F1: 86.20%

---

### Linear SVM

Selected because:

* Handles high-dimensional sparse text efficiently.
* Strong generalization properties.
* Widely regarded as a text classification benchmark.

Final Performance:

* Accuracy: 85.59%
* Macro F1: 84.18%

---

## Probabilistic Models

### Naive Bayes

Selected because:

* Strong historical performance on text classification.
* Extremely efficient training.

Final Performance:

* Accuracy: 66.04%
* Macro F1: 64.47%

---

## Distance-Based Models

### KNN

Selected as a non-parametric baseline.

Final Performance:

* Accuracy: 52.68%
* Macro F1: 49.37%

The model struggled due to the curse of dimensionality in sparse feature spaces.

---

## Tree-Based Boosting Models

### XGBoost

Final Performance:

* Accuracy: 83.36%
* Macro F1: 82.03%

### LightGBM

Final Performance:

* Accuracy: 83.08%
* Macro F1: 81.86%

These models successfully captured nonlinear interactions between engineered features.

---

## Transformer Models

### RoBERTa Base + Optuna

Selected because:

* Captures contextual semantics.
* Understands word ordering.
* Models long-range dependencies.

Final Performance:

* Accuracy: 88.87%
* Macro F1: 88.13%
* Weighted F1: 88.93%

This model achieved the highest score across all evaluation metrics.

---

# Why Macro F1 Was Used

The dataset exhibits class imbalance.

Accuracy alone can be misleading because it disproportionately reflects performance on majority classes.

Macro F1 was selected because:

* Every class contributes equally.
* Poor minority-class performance is penalized.
* It provides a more reliable estimate of balanced classification quality.

For this reason, model selection and Optuna optimization primarily focused on maximizing Macro F1 Score.

---

# Key Findings

1. Feature engineering contributed more improvement than changing algorithms alone.

2. TF-IDF n-gram optimization produced a substantial performance increase over baseline models.

3. Hyperparameter optimization consistently improved all major evaluation metrics.

4. Linear models remained highly competitive despite the availability of more complex architectures.

5. RoBERTa achieved the best overall performance due to its contextual language understanding capabilities.

6. Macro F1 proved to be the most informative metric for comparing models under class imbalance.

---

# Generated Artifacts

The pipeline generates:

```text
model.pkl
vectorizer.pkl
label_encoder.pkl
optuna_study.pkl
```

MLflow logs:

```text
mlruns/
```

Evaluation artifacts:

```text
confusion_matrix.png
classification_report.txt
feature_importance.csv
optuna_trials.csv
```

---

# Reproducing Experiments

## Clone Repository

```bash
git clone https://github.com/<Sohamgaonkhadkar>/<youtube-sentiment-model-training>.git
cd <repository-name>
```

## Create Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Start MLflow

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

## Execution Order

```text
1. yt_comment_analyzer_preprocessing.ipynb
2. experiment_1_baseline_model.ipynb
3. experiment_2_bow_tfidf.ipynb
4. experiment_3_tfidf_(1,3)_max_features.ipynb
5. experiment_4_handling_imbalanced_data.ipynb
6. custom_features.ipynb
7. experiment_5_lor_with_hpt.ipynb
8. experiment_5_naive_bayes_with_hpt.ipynb
9. experiment_5_knn_with_hpt.ipynb
10. experiment_5_svm_with_hpt.ipynb
11. experiment_5_linearsvm_with_hpt.ipynb
12. experiment_5_random_forest_with_hpt.ipynb
13. experiment_5_xgboost_with_hpt.ipynb
14. experiment_5_lightgbm_with_hpt.ipynb
15. experiment_5_catboost_with_hpt.ipynb
16. experiment_5_roberta_base_with_optuna_hpt.ipynb
```

---

# Conclusion

This repository demonstrates a complete machine learning experimentation lifecycle for large-scale text classification. Through systematic preprocessing, advanced TF-IDF engineering, imbalance mitigation, hyperparameter optimization, and transformer fine-tuning, the project achieved a best Macro F1 Score of 88.13% using RoBERTa Base with Optuna optimization.

The results illustrate the importance of feature engineering, rigorous experimentation, and metric-driven model selection when building high-performance NLP systems.

# Synthetic Tabular Data Generation with Scikit-Learn and SDV

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This project demonstrates synthetic tabular data generation using Synthetic Data Vault (SDV) library and evaluates machine learning model performance on both original and synthetic datasets.

## Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Project Description

This project implements a complete pipeline for:
1. Loading and preprocessing the KDD Cup '99 network intrusion detection dataset
2. Generating synthetic data using four methods:
   - Gaussian Copula
   - CTGAN (Conditional Tabular GAN)
   - Copula GAN
   - TVAE (Tabular Variational Auto-encoder)
3. Evaluating machine learning models (Random Forest, Naive Bayes, Logistic Regression) on:
   - Original data only
   - Original + synthetic data combinations
4. Comparing model performance across different data scenarios

## Features

- **Data Processing**
  - Automated loading and preprocessing of KDD Cup '99 dataset
  - Handling of categorical and numerical features
  - Class balancing for rare categories

- **Synthetic Data Generation**
  - Gaussian Copula implementation
  - CTGAN implementation
  - Synthetic data quality evaluation

- **Model Evaluation**
  - Multiple classifier support
  - Comprehensive metrics (F1, Precision, Recall, Accuracy)
  - Confusion matrix visualization
  - Performance comparison across datasets

## Installation

1. Clone the repository:
```bash
git clone https://github.com/thealper2/synthetic-data-generation.git
cd synthetic-data-generation
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/active # On Windows use `venv\Scripts\activate`
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Download the KDD CUP 99 dataset and update the path in `config.py`:

```python
KDD_DATASET_PATH = "path/to/your/kddcup.data_10_percent"
```

2. Run the main pipeline:

```bash
python3 main.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
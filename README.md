# A real-time machine learning computable phenotype for automated iatrogenic withdrawal syndrome detection in critically ill children using electronic health record data

This repository contains a complete and reproducible workflow for training, calibrating, and evaluating an XGBoost classifier on a clinical time-series dataset.  
All sensitive variable names, lab values, medication names, and private details have been removed for public release.

## Features

This pipeline includes:

- Data preprocessing & scaling
- XGBoost model training with class imbalance handling
- Sigmoid (Platt) calibration of predicted probabilities
- Evaluation metrics including:
  - AUROC, AUPRC
  - Sensitivity, Specificity
  - F1-score, MCC, PPV, NPV
- ROC curve, precision–recall curve
- Calibration curve
- Feature importance (top-10)
- Clean modular structure, suitable for research pipelines

## Project Structure

```
Computable-Phenotype/
│
├── src/
│   ├── train_xgb_model.py        # The cleaned script
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

Requires Python ≥ 3.11.

```
git clone https://github.com/haoqiusong/Computable-Phenotype.git
cd Computable-Phenotype
pip install -r requirements.txt
```

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

## 📂 Project Structure

```
your-repo-name/
│
├── src/
│   ├── train_xgb_model.py        # The cleaned/polished script
│   ├── utils.py                  # (Optional) Utilities if needed later
│
├── data/
│   ├── your_dataset.csv          # (Private, do NOT upload to GitHub)
│
├── models/
│   ├── README.md                 # (Placeholder)
│
├── results/
│   ├── plots/                    # ROC, PR, Calibration, FI
│   ├── metrics/                  # JSON/CSV performance outputs
│
├── requirements.txt
├── README.md
└── LICENSE                       # Recommend MIT license
```

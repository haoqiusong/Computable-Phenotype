import graphviz
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import joblib


# Data Preprocessing

# Load the dataset
df = pd.read_csv('data.csv')

# Remove datapoints without recorded WAT scores
missing_data_df = df[df['Max WAT score'] == -1]
df = df[df['Max WAT score'] != -1]

patient_ids = df['Patient']
epoch_start_times = df['Epoch Start Time']

# Remove non-feature columns
df = df.drop(columns=['Lactate', 'Glucose', 'pH', 'pCO2', 'pO2', 'Bicarbonate', 'Base Deficit',
                      'WBC', 'Hemoglobin', 'Hematocrit', 'Sodium', 'Potassum', 'Chloride', 'BUN',
                      'Creatinine', 'Calcium', 'Total Protein', 'Albumin', 'Bilirubin Total',
                      'ALT', 'aPTT', 'Absolute Neutrophil Count', 'Base Excess', 'PaO2',
                      'Fibrinogen', 'Bilirubin Conjugated', 'Methemoglobin', 'Bilirubin Indirect',
                      'AST', 'PT', 'Epoch End Time',
                      'dilaudid master (HYDROmorphone) (Existing bolus dose)', 'dilaudid master (HYDROmorphone) (Dosage count)',
                      'fentaNYL (Existing bolus dose)', 'fentaNYL (Dosage count)',
                      'LORazepam (Existing bolus dose)', 'LORazepam (Dosage count)',
                      'methadone (Existing bolus dose)', 'methadone (Dosage count)',
                      'midazolam (Existing bolus dose)', 'midazolam (Dosage count)',
                      'morphine (Existing bolus dose)', 'morphine (Dosage count)',
                      'propofol (Existing bolus dose)', 'propofol (Dosage count)',
                      'dilaudid master (HYDROmorphone) (Increase percentage)', 'fentaNYL (Increase percentage)', 'LORazepam (Increase percentage)', 
                      'methadone (Increase percentage)', 'midazolam (Increase percentage)', 'morphine (Increase percentage)', 
                      'propofol (Increase percentage)', 'dexmedetomidine (Increase percentage)',
                      'dexmedetomidine (Existing bolus dose)', 'dexmedetomidine (Dosage count)',
                      'Temperature (Standard deviation)', 'Heart Rate (Standard deviation)', 'Systolic (Standard deviation)', 'Diastolic (Standard deviation)',
                      'Respiratory Rate (Standard deviation)', 'Mean Arterial Pressure (Standard deviation)'])

df = df.drop(columns=['Any Loose/Watery Stools WAT', 'Any Vomiting/Wretching/Gagging WAT', 'Temp Often >37.8 WAT', 'State WAT', 'Tremor WAT', 'Any Sweating WAT', 'Uncoordinated/Repetitive Movements WAT', 'Yawning or Sneezing >1 WAT', 'Startle to Touch WAT', 'Muscle tone WAT', 'Time to gain calm WAT'])

df = df.dropna(axis=1, how='all')

# Define the columns to exclude from training (including '(Dosage and time)' columns)
columns_to_exclude = ['Patient', 'Label', 'Epoch Start Time', 'Max WAT score',
                      'dilaudid master (HYDROmorphone) (Dosage and time)', 'fentaNYL (Dosage and time)',
                      'LORazepam (Dosage and time)', 'methadone (Dosage and time)',
                      'midazolam (Dosage and time)', 'morphine (Dosage and time)',
                      'propofol (Dosage and time)', 'dexmedetomidine (Dosage and time)']

# Define features and target variable
X = df.drop(columns=columns_to_exclude, errors='ignore')
y = df['Label']
max_wat_scores = df['Max WAT score']

# Keep the '(Dosage and time)' columns for later use
dosage_time_cols = ['dilaudid master (HYDROmorphone) (Dosage and time)', 'fentaNYL (Dosage and time)',
                    'LORazepam (Dosage and time)', 'methadone (Dosage and time)',
                    'midazolam (Dosage and time)', 'morphine (Dosage and time)',
                    'propofol (Dosage and time)', 'dexmedetomidine (Dosage and time)']

dosage_time_data = df[dosage_time_cols + ['Patient', 'Epoch Start Time']]

# Perform train-test split
X_train, X_test, y_train, y_test, max_wat_train, max_wat_test, patient_train, patient_test, epoch_start_train, epoch_start_test, dosage_time_train, dosage_time_test = train_test_split(
    X, y, max_wat_scores, patient_ids, epoch_start_times, dosage_time_data, test_size=0.2, stratify=y, random_state=42)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Calculate the scale_pos_weight
count_0 = (y_train == 0).sum()
count_1 = (y_train == 1).sum()
ratio = count_0 / count_1 if count_1 > 0 else 1


# Model Training & Testing

# Initialize and train the XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=ratio,
                      learning_rate=0.01, n_estimators=500, max_depth=10, max_delta_step=1.5)
model.fit(X_train, y_train)

# Predictions
y_pred_proba1 = model.predict_proba(X_test)[:, 1]
threshold = 0.5
y_pred = (y_pred_proba1 >= threshold).astype(int)

# Performance Evaluation

# Evaluation metrics
auroc = roc_auc_score(y_test, y_pred_proba1)
auprc = average_precision_score(y_test, y_pred_proba1)
mcc = matthews_corrcoef(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)
specificity = recall_score(y_test, y_pred, pos_label=0)
f1 = f1_score(y_test, y_pred)
ppv = precision_score(y_test, y_pred)
npv = precision_score(y_test, y_pred, pos_label=0)

# Confusion Matrix for calculating specificity and NPV manually if needed
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity_manual = tn / (tn + fp)
npv_manual = tn / (tn + fn)

# Print the results
print(f"AUROC: {auroc}")
print(f"AUPRC: {auprc}")
print(f"MCC: {mcc}")
print(f"Accuracy: {accuracy}")
print(f"Sensitivity (Recall): {sensitivity}")
print(f"Specificity: {specificity}")
print(f"F1-Score: {f1}")
print(f"PPV (Precision): {ppv}")
print(f"NPV: {npv}")

# Save the model and scaler
joblib.dump(model, 'xgboost_model.pkl')
joblib.dump(scaler, 'scaler.pkl')


# Model Evaluation

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_pred_proba1)
roc_auc = auc(fpr, tpr)

# Compute Precision-Recall curve and area
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba1)
pr_auc = auc(recall, precision)

# Plotting the ROC Curve
plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc.png')

# Plotting the Precision-Recall Curve
plt.figure(figsize=(7, 7))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.0])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig('prc.png')

# Visualize the decision tree
booster = model.get_booster()
booster.feature_names = X.columns.tolist()

dot_data = xgb.to_graphviz(booster, num_trees=0, rankdir='LR')
dot_data.render(filename='tree_visualization', format='png', cleanup=True)

# Visualize the feature importance
feature_names = X.columns.tolist()
importances = model.feature_importances_
features = np.array(feature_names)
indices = np.argsort(importances)[::-1]
# Reorder feature names so they match the sorted importances
sorted_feature_names = features[indices]
# Reorder importances
sorted_importances = importances[indices]
# Plotting feature importances
plt.figure(figsize=(38, 20))
plt.barh(range(len(sorted_importances)), sorted_importances[::-1], color='b', align='center')
plt.yticks(range(len(sorted_importances)), sorted_feature_names[::-1])
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Feature Importances')
plt.savefig('feature_importance.png')
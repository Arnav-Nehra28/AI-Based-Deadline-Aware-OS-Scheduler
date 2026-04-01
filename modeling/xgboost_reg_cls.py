import os
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix
)

# =====================================
# PATHS
# =====================================

DATA_PATH = "./data"
RESULT_PATH = "./results/XGBoost_result"

os.makedirs(RESULT_PATH, exist_ok=True)

# =====================================
# LOAD DATA
# =====================================

print("\nLoading datasets...")

train_df = pd.read_csv(os.path.join(DATA_PATH, "processed_train.csv"), low_memory=False)
val_df = pd.read_csv(os.path.join(DATA_PATH, "processed_val.csv"), low_memory=False)
test_df = pd.read_csv(os.path.join(DATA_PATH, "processed_test.csv"), low_memory=False)

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)

# =====================================
# FEATURE ENGINEERING
# =====================================

def create_features(df):
    df["cpu_ratio"] = df["cpu_task"] / df["cpu_machine"].replace(0, np.nan)
    df["mem_ratio"] = df["mem_task"] / df["mem_machine"].replace(0, np.nan)
    df["disk_ratio"] = df["disk_task"] / df["disk_machine"].replace(0, np.nan)

    df["cpu_gap"] = df["cpu_machine"] - df["cpu_task"]
    df["mem_gap"] = df["mem_machine"] - df["mem_task"]

    df["resource_pressure"] = (
        df["cpu_ratio"] +
        df["mem_ratio"] +
        df["disk_ratio"]
    )

    df["task_hour"] = (df["start_time"] % 86400) // 3600

    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    return df


train_df = create_features(train_df)
val_df = create_features(val_df)
test_df = create_features(test_df)

# =====================================
# REMOVE USELESS FEATURES
# =====================================

for df in [train_df, val_df, test_df]:
    if "instance_id" in df.columns:
        df.drop("instance_id", axis=1, inplace=True)

    if "end_time" in df.columns:
        df.drop("end_time", axis=1, inplace=True)

# =====================================
# FEATURE / TARGET SPLIT
# =====================================

feature_cols = [c for c in train_df.columns if c not in ["machine_id", "duration"]]

X_train = train_df[feature_cols].astype(np.float32)
X_val = val_df[feature_cols].astype(np.float32)
X_test = test_df[feature_cols].astype(np.float32)

y_train_clf = train_df["machine_id"]
y_val_clf = val_df["machine_id"]

y_train_reg = train_df["duration"]
y_val_reg = val_df["duration"]

# =====================================
# LABEL ENCODING
# =====================================

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train_clf)

known_classes = set(label_encoder.classes_)

mask = y_val_clf.isin(known_classes)

X_val_filtered = X_val[mask]
y_val_filtered = label_encoder.transform(y_val_clf[mask])

num_classes = len(label_encoder.classes_)

print("\nMachine Classes:", num_classes)

# =====================================
# XGBOOST CLASSIFIER (GPU)
# =====================================

print("\nTraining XGBoost Classifier...")

xgb_clf = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=num_classes,
    learning_rate=0.05,
    max_depth=4,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    sampling_method="gradient_based",
    gamma=0.1,
    min_child_weight=5,
    reg_alpha=0.5,
    reg_lambda=1.5,
    tree_method="hist",
    device="cuda",
    max_bin=256,
    eval_metric="mlogloss",
    early_stopping_rounds=20,
    n_jobs=-1,
    random_state=42
)

start = time.time()

xgb_clf.fit(
    X_train,
    y_train_encoded,
    eval_set=[(X_val_filtered, y_val_filtered)],
    verbose=True
)

training_time_clf = time.time() - start

# =====================================
# CLASSIFICATION PREDICTIONS
# =====================================

pred_encoded = xgb_clf.predict(X_val_filtered)

pred_labels = label_encoder.inverse_transform(pred_encoded)

accuracy = accuracy_score(y_val_clf[mask], pred_labels)

precision = precision_score(
    y_val_clf[mask],
    pred_labels,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_val_clf[mask],
    pred_labels,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_val_clf[mask],
    pred_labels,
    average="weighted",
    zero_division=0
)

# =====================================
# TOP-5 ACCURACY
# =====================================

probs = xgb_clf.predict_proba(X_val_filtered)

top5 = np.argsort(probs, axis=1)[:, -5:]

true_labels = label_encoder.transform(y_val_clf[mask])

top5_acc = np.mean([
    true_labels[i] in top5[i]
    for i in range(len(true_labels))
])

print("Top-5 Accuracy:", top5_acc)

# =====================================
# CONFUSION MATRIX
# =====================================

print("\nGenerating Confusion Matrix...")

top_classes = y_val_clf.value_counts().index[:20]

top_mask = y_val_clf[mask].isin(top_classes)

pred_labels_filtered = pred_labels[top_mask.values]

cm = confusion_matrix(
    y_val_clf[mask][top_mask],
    pred_labels_filtered
)

plt.figure(figsize=(12, 10))

sns.heatmap(cm, cmap="Blues")

plt.title("Confusion Matrix (Top 20 Machines)")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(os.path.join(RESULT_PATH, "confusion_matrix.png"))

plt.close()

# =====================================
# XGBOOST REGRESSOR
# =====================================

print("\nTraining XGBoost Regressor...")

y_train_log = np.log1p(y_train_reg)
y_val_log = np.log1p(y_val_reg)

xgb_reg = xgb.XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.05,
    max_depth=4,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    sampling_method="gradient_based",
    tree_method="hist",
    device="cuda",
    max_bin=256,
    eval_metric="rmse",
    early_stopping_rounds=20,
    n_jobs=-1,
    random_state=42
)

start = time.time()

xgb_reg.fit(
    X_train,
    y_train_log,
    eval_set=[(X_val, y_val_log)],
    verbose=True
)

training_time_reg = time.time() - start

# =====================================
# REGRESSION METRICS
# =====================================

pred_log = xgb_reg.predict(X_val)

pred_reg = np.expm1(pred_log)

mae = mean_absolute_error(y_val_reg, pred_reg)

mse = mean_squared_error(y_val_reg, pred_reg)

rmse = np.sqrt(mse)

r2 = r2_score(y_val_reg, pred_reg)

# =====================================
# SAVE MODELS
# =====================================

joblib.dump(
    xgb_clf,
    os.path.join(RESULT_PATH, "xgb_classifier.pkl")
)

joblib.dump(
    xgb_reg,
    os.path.join(RESULT_PATH, "xgb_regressor.pkl")
)

# =====================================
# RESULTS
# =====================================

results = {
    "Accuracy": accuracy,
    "Top5_Accuracy": top5_acc,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "R2": r2,
    "Training Time Classifier": training_time_clf,
    "Training Time Regressor": training_time_reg
}

pd.DataFrame([results]).to_csv(
    os.path.join(RESULT_PATH, "metrics.csv"),
    index=False
)

print("\nTraining Completed\n")

for k, v in results.items():
    print(k, ":", v)

print("\nResults saved to:", RESULT_PATH)

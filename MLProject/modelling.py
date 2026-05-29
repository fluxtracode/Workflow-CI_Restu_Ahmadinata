import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import mlflow
import mlflow.sklearn
import os
import shutil

# 1. Data Loading & Splitting
df = pd.read_csv('churn_preprocessing.csv')
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Setup Model & Hyperparameter Tuning
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='f1_macro')

# 3. Memulai Tracking di CI Pipeline
# HAPUS mlflow.set_experiment() agar tidak bentrok dengan 'mlflow run'

# UBAH menjadi mlflow.start_run() kosong agar ia otomatis menyambung ke Run yang sudah dibuat
with mlflow.start_run():
    # Fit model
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Prediksi
    y_pred = best_model.predict(X_test)
    
    # Hitung Metrik
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # --- MANUAL LOGGING ---
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    
    # --- ARTEFAK ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png') 
    
    report = classification_report(y_test, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact('classification_report.txt') 
    
    # --- PENYIMPANAN MODEL UNTUK DOCKER ---
    model_dir = "saved_model"
    
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        
    mlflow.sklearn.save_model(best_model, model_dir)
    print("Training selesai! Model disimpan di folder 'saved_model' siap untuk di-build ke Docker.")
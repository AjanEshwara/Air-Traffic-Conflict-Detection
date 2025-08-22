import os
import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier,RandomForestRegressor,
                              GradientBoostingRegressor)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC,SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, mean_squared_error, r2_score, roc_auc_score,
                             root_mean_squared_error,
                             root_mean_squared_log_error, mean_absolute_error, mean_squared_log_error)
import joblib
from imblearn.over_sampling import SMOTE

def train_rf_classifier(train_path, test_path, model_output_path, target_col='SI', random_state=42):
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    #classification
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]



    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    smote = SMOTE(random_state=random_state)
    X_train_smt, y_train_smt = smote.fit_resample(X_train, y_train)
    print(f"SMOTE-applied training data shape: {X_train_smt.shape}")

    # Initialize classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=random_state),
        'XGBoost': XGBClassifier(eval_metric='logloss',class_weight='balanced',random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(class_weight='balanced',random_state=random_state),
        'SVM': SVC(class_weight='balanced',probability=True,random_state=random_state),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(class_weight='balanced',random_state=random_state),

    }

    for name,clf in classifiers.items():
        #train data set
        clf.fit(X_train_smt, y_train_smt)
        #predict probabilities
       # probs = clf.predict_proba(X_test)[:, 1]
        #predict data set with custom threshold
       # threshold = 0.2
        #y_pred = (probs >= threshold).astype(int)
        y_pred = clf.predict(X_test)
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred,digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
        print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
        print(f"auc Score:  {roc_auc_score(y_test, y_pred):.4f}")


    # Save the model to disk classifier
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(clf, model_output_path)
    print(f"Model saved to {model_output_path}")



from sklearn.preprocessing import StandardScaler
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .config import N_FOLDS, RANDOM_STATE
import category_encoders as ce

def evaluate_models(X, y_encoded, models):
   
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    num_models = len(models)
    results = np.zeros((num_models, N_FOLDS, 4))
    
    for i, (train_index, test_index) in enumerate(cv.split(X, y_encoded)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        woe_encoder = ce.TargetEncoder(cols=['domain'])
        scaler = StandardScaler()
       
        print(f"\n--- Fold {i+1} ---")
        
        start_time_enc = time.time()
        X_train_encoded = woe_encoder.fit_transform(X_train, y_train)
        X_test_encoded = woe_encoder.transform(X_test)
        
        X_train_scaled = scaler.fit_transform(X_train_encoded)
        X_test_scaled = scaler.transform(X_test_encoded)
        end_time_enc = time.time()
        print(f"[Fold {i+1}] Encoding & Scaling time: {end_time_enc - start_time_enc:.2f} seconds")

        for j, (model_name, clf) in enumerate(models.items()):
            clf_clone = clone(clf)
            
            start_time_train = time.time()
            clf_clone.fit(X_train_scaled, y_train)
            end_time_train = time.time()
            print(f"[Fold {i+1}] {model_name} training time: {end_time_train - start_time_train:.2f} seconds")
            
            prediction = clf_clone.predict(X_test_scaled)

            acc = accuracy_score(y_test, prediction)
            prec = precision_score(y_test, prediction, average='weighted')
            rec = recall_score(y_test, prediction, average='weighted')
            f1 = f1_score(y_test, prediction, average='weighted')

            results[j, i] = [acc, prec, rec, f1]
            
    return results

def print_results(results, models):
    
    for j, model_name in enumerate(models.keys()):
        mean_scores = results[j].mean(axis=0)
        std_scores = results[j].std(axis=0)

        print(f"{model_name}")
        print(f"Accuracy:  {mean_scores[0]:.4f} (± {std_scores[0]:.4f})")
        print(f"Precision: {mean_scores[1]:.4f} (± {std_scores[1]:.4f})")
        print(f"Recall:    {mean_scores[2]:.4f} (± {std_scores[2]:.4f})")
        print(f"F1 Score:  {mean_scores[3]:.4f} (± {std_scores[3]:.4f})")

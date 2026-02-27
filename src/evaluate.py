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
    evaluation_results = {model_name: [] for model_name in models.keys()}
    
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

            evaluation_results[model_name].append({
                'fold': i + 1,
                'y_true': y_test.tolist(),
                'y_pred': prediction.tolist(),
                'metrics': {
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1
                }
            })
            
    return results, evaluation_results

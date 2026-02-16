
#Imports
import kagglehub
import pandas as pd
import os
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

#Data download
path = kagglehub.dataset_download("sid321axn/malicious-urls-dataset")
print("Path to dataset files:", path)
print(os.listdir(path))

df = pd.read_csv(os.path.join(path, "malicious_phish.csv"))
print(df.head())
print(df.shape)
print(df.info()) # To view the column names
print(f"{df['type'].value_counts(normalize=True) * 100}") #We analyze the distribution of the classes

# Classifiers
clf = XGBClassifier()
clf2 = KNeighborsClassifier()
clf3 = MLPClassifier()

models = {
    'XGBClassifier': clf,
    'KNeighborsClassifier': clf2,
    'MLPClassifier': clf3
}

#Feature engineering

# 1. URL Length
df['url_len'] = df['url'].apply(len)

# 2. Dot Count for detecting subdomains
df['dot_count'] = df['url'].apply(lambda x: x.count('.'))

# 3. Digit Count for detecting random numbers
df['digit_count'] = df['url'].apply(lambda x: sum(c.isdigit() for c in x))

#Label and input assignment
X = df[['url', 'url_len', 'dot_count', 'digit_count']]
y = df['type']

#data encoding

le = LabelEncoder()
y_encoded = le.fit_transform(y)

#Cross validation setup

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

arr = np.zeros((3,5,4))

for i, (train_index, test_index) in enumerate(tqdm(cv.split(X, y), total=5, desc="CV Folds")):

  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y_encoded[train_index], y_encoded[test_index]

  #data encoding part 2

  woe_encoder = ce.TargetEncoder(cols=['url'])
  X_train_encoded = woe_encoder.fit_transform(X_train, y_train)
  X_test_encoded = woe_encoder.transform(X_test)

  for j, (model_name, clf) in enumerate(models.items()):
    clf_clone = clone(clf)
    clf_clone.fit(X_train_encoded, y_train)
    prediction = clf_clone.predict(X_test_encoded)

    acc = accuracy_score(y_test, prediction)
    prec = precision_score(y_test, prediction, average='weighted')
    rec = recall_score(y_test, prediction, average='weighted')
    f1 = f1_score(y_test, prediction, average='weighted')

    arr[j, i] = [acc, prec, rec, f1]

for j, model_name in enumerate(models.keys()):
  mean_scores = arr[j].mean(axis=0)
  std_scores = arr[j].std(axis=0)
  print(f"{model_name}")
  print(f"Accuracy:  {mean_scores[0]:.4f} (± {std_scores[0]:.4f})")
  print(f"Precision: {mean_scores[1]:.4f} (± {std_scores[1]:.4f})")
  print(f"Recall:    {mean_scores[2]:.4f} (± {std_scores[2]:.4f})")
  print(f"F1 Score:  {mean_scores[3]:.4f} (± {std_scores[3]:.4f})")
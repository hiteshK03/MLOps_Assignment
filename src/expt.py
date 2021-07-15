import pandas as pd
import numpy as np
import json
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import dvc.api

with dvc.api.open(repo="https://github.com/hiteshK03/MLOps_Assignment", path="data/creditcard.csv", mode="r") as fd:
    df = pd.read_csv(fd)

# df = pd.read_csv("data/creditcard.csv")
# print(df)

# y = df['Class']
# print(y)
stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
target = 'Class'
predictors = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',\
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',\
       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',\
       'Amount']

X, y = df[predictors], df[target].values
for train_idx, test_idx in stratSplit.split(X,y):
    X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
    y_train, y_test = y[train_idx], y[test_idx]

X_train["Class"] = y_train
X_test["Class"] = y_test

path = "./data/processed/"
X_train.to_csv(path+"train.csv")
X_test.to_csv(path+"test.csv")

clf = RandomForestClassifier(criterion="entropy", random_state=42)
clf.fit(X_train, y_train)

joblib.dump(clf, './models/model.pkl')
preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)

metrics = {
	"accuracy_score" : acc,
	"f1_score" : f1
}
with open("./metrics/acc_f1.json", "w") as outfile: 
    json.dump(metrics, outfile)
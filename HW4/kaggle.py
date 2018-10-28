import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("train_data.csv").drop('Id', axis=1)
train_data_labels = pd.read_csv("train_labels.csv").drop('Id', axis=1)
test_data = pd.read_csv("test_data.csv").drop('Id', axis=1)

# KNN
n_neighbors = 3
knn_model = neighbors.KNeighborsClassifier()
knn_model.fit(train_data, train_data_labels.values.ravel())
class_predict = pd.DataFrame(knn_model.predict(test_data)).to_csv("lr_knn_3.csv")

# Logistic Regression
lr_model = LogisticRegression(solver='lbfgs')
lr_model.fit(train_data, train_data_labels.values.ravel())
class_predict = pd.DataFrame(data=np.argmax(lr_model.predict_log_proba(test_data), axis=1)).to_csv("lr_lbfgs.csv")

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(train_data, train_data_labels.values.ravel())
class_predict = pd.DataFrame(data=np.argmax(nb_model.predict_log_proba(test_data), axis=1)).to_csv("nb.csv")

# SVM
svm_model = svm.LinearSVC()
svm_model.fit(train_data, train_data_labels.values.ravel())
class_predict = pd.DataFrame(data=np.argmax(svm_model.decision_function(test_data), axis=1)).to_csv("lr_svmLinearSVC.csv")

# Random Forests
rf_model = RandomForestClassifier(n_estimators=2000, max_features="auto", max_depth=None)
rf_model.fit(train_data, train_data_labels.values.ravel())
class_predict = pd.DataFrame(data=rf_model.predict(test_data)).to_csv("rf_estimators_2000.csv")

import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

train_data = pd.read_csv("train_data.csv").drop('Id', axis=1)
train_data_labels = pd.read_csv("train_labels.csv").drop('Id', axis=1)

lr_model = LogisticRegression()
lr_model.fit(train_data.head(100), train_data_labels.head(100).values)

test_data = pd.read_csv("test_data.csv").drop('Id', axis=1)

class_predict = pd.DataFrame(data=np.argmax(lr_model.predict_log_proba(test_data), axis=1)).to_csv("temp.csv")

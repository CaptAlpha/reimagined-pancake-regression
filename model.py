import pandas as pd
import numpy as np


import pickle


train_df = pd.read_csv("train_.csv")
df = train_df
df = df.drop(['Unnamed: 0'], axis=1)
label_col = "click"
x_columns = set(list(df.columns)) - set(["id", "site_id", "app_id", "hour", "dt_hour", "device_id", "device_ip", ] + [label_col] )
x_train = df[x_columns]
y_train = df[label_col]
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder


x_train_len = len(x_train)
d = defaultdict(LabelEncoder)
n_df = x_train.apply(lambda x: d[x.name].fit_transform(x))
n_df.head()
X = n_df
y = df['click']
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
model=DecisionTreeClassifier(criterion='entropy')
model.fit(X_train,y_train)

pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
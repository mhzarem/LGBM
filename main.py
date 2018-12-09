import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing the data_set
data_set = pd.read_csv('Churn_Modelling.csv')
X = data_set.iloc[:, 3:13].values
y = data_set.iloc[:, 13].values

# Encoding categorical data
label_encoder_X_1 = LabelEncoder()
X[:, 1] = label_encoder_X_1.fit_transform(X[:, 1])
label_encoder_X_2 = LabelEncoder()
X[:, 2] = label_encoder_X_2.fit_transform(X[:, 2])
one_hot_encoder = OneHotEncoder(categorical_features=[1])
X = one_hot_encoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the data_set into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

d_train = lgb.Dataset(X_train, label=Y_train)

params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 500
params['max_depth'] = 10
params['boost_from_average'] = 'false'

# training the model
clf = lgb.train(params, d_train, 100)

y_pred = clf.predict(X_test)

for i in range(0, len(y_pred)):
    if y_pred[i] >= .5:  # setting threshold to .5
        y_pred[i] = 1
    else:
        y_pred[i] = 0
cm = confusion_matrix(Y_test, y_pred)
print(cm)

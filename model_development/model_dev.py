# -*- coding: utf-8 -*-
"""

"""

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_squared_log_error

df = pd.read_csv('training_data_wImpute_threshold_0.1.csv')

labels = df['meter_reading']
data = df.drop(columns=['meter_reading'])

#print(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3, random_state=42)

reg_tree = tree.DecisionTreeRegressor()
reg_tree = reg_tree.fit(X_train, y_train)
predictions = reg_tree.predict(X_test)

#print(y_test)
#print(predictions)
print(np.sqrt(mean_squared_log_error(y_test,predictions)))

# Note: returns negative values which causes error when calculating RSMLE
linear_reg = LinearRegression(fit_intercept=False).fit(X_train,y_train)
predictions = linear_reg.predict(X_test)

#print(predictions)
#print(np.sqrt(mean_squared_log_error(y_test,predictions)))

knn_reg = KNeighborsRegressor(n_neighbors=3,metric='euclidean').fit(X_train,y_train)
predictions = knn_reg.predict(X_test)

print(predictions)
print(np.sqrt(mean_squared_log_error(y_test,predictions)))

nn = MLPRegressor(activation='logistic').fit(X_train,y_train)
predictions = nn.predict(X_test)

print(predictions)
print(np.sqrt(mean_squared_log_error(y_test,predictions)))


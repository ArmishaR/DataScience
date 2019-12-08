# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score
from yellowbrick.regressor import ResidualsPlot, PredictionError
import matplotlib.pyplot as plt
import seaborn as sns
import time

df = pd.read_csv('training_data_series5/training_data_woEncoding_threshold_0.8.csv')
df = df.drop(columns=df.columns[0], axis=1)

# Encoding
le = preprocessing.LabelEncoder()
le.fit(df['primary_use'])
df['primary_use'] = le.transform(df['primary_use'])

labels = df['meter_reading']
data = df.drop(columns=['meter_reading'])

# Open to print results
f = open('nn_results.txt','a+')

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3, random_state=42)

""" Model Predictions of Test Set """
'''
t = time.process_time()
reg_tree = tree.DecisionTreeRegressor(criterion='mse')
reg_tree = reg_tree.fit(X_train, y_train)
elapsed_train = time.process_time() - t
t = time.process_time()
predictions = reg_tree.predict(X_test)
elapsed_test = time.process_time() - t

rmsle_dt = np.sqrt(mean_squared_log_error(y_test,predictions))
mae_dt = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions, multioutput='variance_weighted')
adj_r2 = 1 - ((1 - r2)*((len(y_test)-1)/(len(y_test)-len(X_test.columns) - 1)))
r2_dt = r2
adj_r2_dt = adj_r2
'''

'''
t = time.process_time()
print("1")
rf = RandomForestRegressor(n_estimators=25, criterion='mse')
print("2")
rf = rf.fit(X_train, y_train)
elapsed_train = time.process_time() - t
t = time.process_time()
print("3")
predictions = rf.predict(X_test)
print("4")
elapsed_test = time.process_time() - t

rmsle_rf = np.sqrt(mean_squared_log_error(y_test,predictions))
mae_rf = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions, multioutput='variance_weighted')
adj_r2 = 1 - ((1 - r2)*((len(y_test)-1)/(len(y_test)-len(X_test.columns) - 1)))
r2_rf = r2
adj_r2_rf =adj_r2
'''


# K-Nearest Neighbors (KNN) Results
'''
t = time.process_time()
knn_reg = KNeighborsRegressor(n_neighbors=8,metric='manhattan').fit(X_train,y_train)
elapsed_train = time.process_time() - t
t = time.process_time()
predictions = knn_reg.predict(X_test)
elapsed_test = time.process_time() - t

rmsle_knn = np.sqrt(mean_squared_log_error(y_test,predictions))
mae_knn = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions, multioutput='variance_weighted')
adj_r2 = 1 - ((1 - r2)*((len(y_test)-1)/(len(y_test)-len(X_test.columns) - 1)))
r2_knn = r2
adj_r2_knn = adj_r2
'''



t = time.process_time()
nn = MLPRegressor(hidden_layer_sizes=(2,),activation='tanh').fit(X_train,y_train)
elapsed_train = time.process_time() - t
t = time.process_time()
predictions = nn.predict(X_test)
elapsed_test = time.process_time() - t


rmsle_nn = np.sqrt(mean_squared_log_error(y_test,predictions))
mae_nn = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions, multioutput='variance_weighted')
adj_r2 = 1 - ((1 - r2)*((len(y_test)-1)/(len(y_test)-len(X_test.columns) - 1)))
r2_nn = r2
adj_r2_nn = adj_r2


'''
print('training time: ', elapsed_train)
print('prediction time: ', elapsed_test)
print('test RMSLE = ',np.mean(rmsle_dt))
print('test R^2 = ',np.mean(r2_dt))
print('test Adj R^2 = ',np.mean(adj_r2_dt))
print('test MAE = ',np.mean(mae_dt))

f.write('Decision Tree' + '\n')
f.write('training time: ' + str(elapsed_train) + 's \n')
f.write('prediction time: ' + str(elapsed_test) + 's \n')
f.write('test RMSLE = ' + str(rmsle_dt) +'\n')
f.write('test R^2 = ' + str(r2_dt) +'\n')
f.write('test Adj R^2 = '+ str(adj_r2_dt) +'\n')
f.write('test MAE = ' + str(mae_dt) +'\n')
'''

'''
print('training time: ', elapsed_train)
print('prediction time: ', elapsed_test)
print('test RMSLE = ',rmsle_rf)
print('test R^2 = ', r2_rf)
print('test Adj R^2 = ', adj_r2_rf)
print('test MAE = ', mae_rf)

f.write('Random Forest' + '\n')
f.write('training time: ' + str(elapsed_train) + 's \n')
f.write('prediction time: ' + str(elapsed_test) + 's \n')
f.write('test RMSLE = ' + str(rmsle_rf) +'\n')
f.write('test R^2 = ' + str(r2_rf) +'\n')
f.write('test Adj R^2 = '+ str(adj_r2_rf) +'\n')
f.write('test MAE = ' + str(mae_rf) +'\n')
'''

'''
print('training time: ', elapsed_train)
print('prediction time: ', elapsed_test)
print('test RMSLE = ', rmsle_knn)
print('test R^2 = ', r2_knn)
print('test Adj R^2 = ', adj_r2_knn)
print('test MAE = ', mae_knn)

f.write('KNN' + '\n')
f.write('training time: ' + str(elapsed_train) + 's \n')
f.write('prediction time: ' + str(elapsed_test) + 's \n')
f.write('test RMSLE = ' + str(rmsle_knn) +'\n')
f.write('test R^2 = ' + str(r2_knn) +'\n')
f.write('test Adj R^2 = '+ str(adj_r2_knn) +'\n')
f.write('test MAE = ' + str(mae_knn) +'\n')
'''


print('training time: ', elapsed_train)
print('prediction time: ', elapsed_test)
print('test RMSLE = ', rmsle_nn)
print('test R^2 = ', r2_nn)
print('test Adj R^2 = ', adj_r2_nn)
print('test MAE = ', mae_nn)

f.write('Multi-Layer Perceptron' + '\n')
f.write('training time: ' + str(elapsed_train) + 's \n')
f.write('prediction time: ' + str(elapsed_test) + 's \n')
f.write('test RMSLE = ' + str(rmsle_nn) +'\n')
f.write('test R^2 = ' + str(r2_nn) +'\n')
f.write('test Adj R^2 = '+ str(adj_r2_nn) +'\n')
f.write('test MAE = ' + str(mae_nn) +'\n')


f.close()

'''sns.set(style="darkgrid")
ax = sns.distplot(predictions)
plt.show()

ax = sns.distplot(y_test)
plt.show()'''
plt.hist(predictions, 50, facecolor='g', alpha=0.75, log=True)
plt.hist(y_test, 50, facecolor='b', alpha=0.5, log=True)
plt.title("Comparison of true and predicted meter readings")
plt.show()

plt.subplot(2, 1, 1)
plt.hist(predictions, 50, facecolor='g', alpha=0.75, log=True)
plt.title("Predicted Meter Readings")

plt.subplot(2, 1, 2)
plt.hist(y_test, 50, facecolor='b', alpha=0.5, log=True)
plt.title("True Meter Readings")
plt.show()



visualizer = ResidualsPlot(nn)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()

visualizer = PredictionError(nn)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()

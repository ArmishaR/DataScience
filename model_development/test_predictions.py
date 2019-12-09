import pandas as pd
import numpy as np
import time

from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score

from yellowbrick.regressor import ResidualsPlot, PredictionError
import matplotlib.pyplot as plt
import seaborn as sns

# this is our final training data (the commented-out is the train data w/o imputation of NaNs)
print("here")
train = pd.read_csv('train_no_na.csv')#pd.read_csv(r'final_wo_imputation\train.csv')
print("here")
train = train.drop(columns=train.columns[0:1], axis=1)
print("here")
print(train.head().mean())

# Encoding
le = preprocessing.LabelEncoder()
le.fit(train['primary_use'])
train['primary_use'] = le.transform(train['primary_use'])

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

# uncomment this code if you want to use another train file - it removes all NaNs
'''imp_mean.fit(train.to_numpy())
print(train.head())
train = pd.DataFrame(imp_mean.transform(train.to_numpy()), columns=train.columns)
print(train.head())
#train.fillna(train.mean(),inplace=True)
print("here")
train.to_csv('train_no_na.csv', index=False)'''

# get labels & data
labels = train['meter_reading']
data = train.drop(columns=['meter_reading'])
print(data.head())
print(data.columns)
print(labels.head())

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3, random_state=42)

# train our RF model
t = time.process_time()
rf = RandomForestRegressor(n_estimators=25, criterion='mse')
rf = rf.fit(X_train, y_train)
elapsed_train = time.process_time() - t
t = time.process_time()
predictions = rf.predict(X_test)
elapsed_test = time.process_time() - t

# evaluate RF model
rmsle_rf = np.sqrt(mean_squared_log_error(y_test,predictions))
mae_rf = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions, multioutput='variance_weighted')
adj_r2 = 1 - ((1 - r2)*((len(y_test)-1)/(len(y_test)-len(X_test.columns) - 1)))
r2_rf = r2
adj_r2_rf =adj_r2

# Print train results to file
f = open('final_results.txt','a+')

print('TRAINING RESULTS')
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

# this is our test data (the commented-out is the test data w/o imputation of NaNs)
test = pd.read_csv('test_no_na.csv')#pd.read_csv(r'final_wo_imputation\test.csv')
ids = test['row_id']
print(test.head())
print(test.columns)

# uncomment this code if you want to use another train file - it removes all NaNs and performs encoding
'''
# Encoding
le = preprocessing.LabelEncoder()
le.fit(test['primary_use'])
test['primary_use'] = le.transform(test['primary_use'])
imp_mean.fit(test.to_numpy())
test = pd.DataFrame(imp_mean.transform(test.to_numpy()), columns=test.columns)
test.to_csv('test_no_na.csv', index=False)
'''
test = test.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'row_id', 'Unnamed: 0.1.1'])
print(test.head())
print(test.columns)
# get predictions
t = time.process_time()
predictions = rf.predict(test)
elapsed_test = time.process_time() - t

print(ids.head())
print(test.head())
print(test.columns)

preds = pd.DataFrame(predictions, columns=['meter_reading'])
preds_w_ids = pd.concat([ids, preds], axis=1, sort=False)
preds_w_ids.to_csv('test_results.csv', index=False)

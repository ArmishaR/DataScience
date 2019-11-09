import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import datetime
from missingpy import KNNImputer
import warnings
import scipy.sparse as ssp
import pydataframe
#warnings.filterwarnings("ignore")

def sparse_ohe(df, col, vals):
    """One-hot encoder using a sparse ndarray."""
    colaray = df[col].values
    # construct a sparse matrix of the appropriate size and an appropriate,
    # memory-efficient dtype
    spmtx = ssp.dok_matrix((df.shape[0], vals.shape[0]), dtype=np.uint8)
    # do the encoding
    spmtx[np.where(colaray.reshape(-1, 1) == vals.reshape(1, -1))] = 1

    # Construct a SparseDataFrame from the sparse matrix
    dfnew = pd.SparseDataFrame(spmtx, dtype=np.uint8, index=df.index,
                               columns=[col + '_' + str(el) for el in vals])
    dfnew.fillna(0, inplace=True)
    print('Finished with ', col)
    return dfnew

# Read in the datasets
#data_info = pd.read_csv('data_info.csv')
train_df = pd.read_csv('train_df_step1.csv')

# Cast cols as a string
train_df.site_id = train_df.site_id.astype(str)
train_df.building_id = train_df.building_id.astype(str)
train_df.meter = train_df.meter.astype(str)

# Code adopted from: https://stackoverflow.com/questions/44228680/pd-get-dummies-slow-on-large-levels
train_df_extended = train_df
for col in ['site_id', 'primary_use', 'building_id', 'meter']:
    vals_train = set(train_df[col].unique())
    #print(np.array(vals_train))
    #vals_test = set(test_df[col].unique())
    #vals = np.array(list(vals_train.union(vals_test)), dtype=np.uint16)
    df_train_temp = sparse_ohe(train_df, col, np.array(list(vals_train)))
    train_df_extended = pd.concat([train_df_extended, df_train_temp], axis=1, sort=False)
    print(train_df_extended.shape)
    #df_test_temp = sparse_ohe(test_df, col, vals)

np.savetxt(r'train_df_extended.txt', train_df_extended, fmt='%s', header=','.join(train_df_extended.columns))
#pydataframe.DF2CSV().write(train_df_extended, 'train_df_extended.csv', dialect=pydataframe.TabDialect())
#train_df_extended.to_hdf(r'train_df_extended.h5', key='default', mode='w', data_columns=True)
#train_df_extended.to_csv('train_df_extended.csv.gz', compression='gzip', chunksize=1000)

train_df_extended = train_df_extended.drop(['site_id', 'primary_use', 'building_id', 'meter'], axis = 1)
var_missing = train_df.isna().sum()
var_percentMissing = var_missing / len(train_df)

print("Looping through thresholds.")

old_count = -1
for t in np.arange(1,0,-0.1):
    # find the variables with missing cells above the threshold
    print("Current threshold: ", t)
    cols = var_percentMissing.index[var_percentMissing>t].tolist()
    count = len(cols)
    if (count == old_count):
        continue
    temp_df = train_df_extended.drop(cols, axis = 1)
    print('Dropped columns')
    
    colnames = temp_df.columns.values
    
    # Normalize and print to a file
    # Impute with KNN
    imputer = KNNImputer(n_neighbors=8)
    temp_df = pd.DataFrame(imputer.fit_transform(temp_df))
    temp_df.columns=colnames
    print('Finished KNN impute')
    
    # Normalize data
    min_max_scaler = MinMaxScaler()
    temp_df = min_max_scaler.fit_transform(temp_df)
    temp_df = pd.DataFrame(temp_df)
    temp_df.columns = colnames
    print('Finished Normalization and printing')
    
    old_count = count
    name = r'training_data_threshold_' + str(t) + '.txt'
    np.savetxt(name, temp_df, fmt='%s', header=','.join(temp_df.columns))


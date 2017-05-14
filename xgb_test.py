
# coding: utf-8

# In[1]:

## authored by wuyi

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
import datetime
dt = datetime.datetime.now()


# In[2]:

def rmspe(y, yhat):
    # y = true data
    # yhat = predication
    return np.sqrt(np.mean(((y - yhat) / y)**2))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)


# In[3]:

monthdic = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

def promo2_indicator(row):
    if row['PromoInterval'] is np.NaN:
        return 0
    try:
        if monthdic[row['Month']-1] == row['PromoInterval']:
            return 1
    except:
        if monthdic[row['Month']-1] in row['PromoInterval']:
            return 1
    return 0


# In[4]:

def build_features(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    
    features.extend(['Store', 'CompetitionDistance', 'Promo', 'Promo2SinceWeek',
                     'Promo2SinceYear', 'SchoolHoliday', 'StoreType', 'Assortment',
                    'StateHoliday'])
    
    mapping = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    data.StoreType.replace(mapping, inplace=True)
    data.Assortment.replace(mapping, inplace=True)
    data.StateHoliday.replace(mapping, inplace=True)
    
    features.extend(['Month', 'Day', 'Year', 'Quarter','DayOfWeek'])
    data['Month'] = data.Date.apply(lambda x: x.month)
    data['Day'] = data.Date.apply(lambda x: x.day)
    data['Year'] = data.Date.apply(lambda x: x.year)
    data['Quarter'] = data.Date.apply(lambda x: x.quarter)
    
    features.extend(['Promo2Indicator', 'CompetitionTime'])
    data['Promo2Indicator'] = data.apply(promo2_indicator, axis = 1)
    data['CompetitionTime'] = data.Month - data.CompetitionOpenSinceMonth + (data.Year - data.CompetitionOpenSinceYear) * 12


# In[5]:

# main script

print("Load the training, test and store data using pandas")
types = {'CompetitionOpenSinceYear': np.dtype(int),
         'CompetitionOpenSinceMonth': np.dtype(int),
         'StateHoliday': np.dtype(str),
         'Promo2SinceWeek': np.dtype(int),
         'SchoolHoliday': np.dtype(int),
         'PromoInterval': np.dtype(str),
         'CompetitionDistance':np.dtype(float),
         'StoreType':np.dtype(str),
         'Assortment':np.dtype(str)}
store = pd.read_csv("../data/store.csv")
test = pd.read_csv("../data/test.csv", parse_dates=[3], dtype = types)
train = pd.read_csv("../data/train.csv", parse_dates=[2], dtype = types)
test = pd.merge(test, store, on = "Store", how = "left")
train = pd.merge(train, store, on = "Store", how = "left")
train = train[train.Sales > 0]

features = []
print("build train features")
build_features(features, train)
print("build test features")
build_features([], test)
print(features)



# In[8]:

params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.01,
          "max_depth": 6,
          "subsample": 1,
          "colsample_bytree": 0.5,
          "silent": 1,
          "seed": 1301
          }
num_boost_round = 7000

# start training
print("Train a XGBoost model")
X_train, X_valid = train_test_split(train, test_size=0.012, random_state=10)
y_train = np.log1p(X_train.Sales)
y_valid = np.log1p(X_valid.Sales)
dtrain = xgb.DMatrix(X_train[features], y_train)
dvalid = xgb.DMatrix(X_valid[features], y_valid)

#X_train = train
#y_train = np.log1p(X_train.Sales)
#dtrain = xgb.DMatrix(X_train[features], y_train)


# In[ ]:

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
#watchlist = [(dtrain, 'train')]

gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,   early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=True)

# predict
print("Make predictions on the test set")
dtest = xgb.DMatrix(test[features])
test_probs = gbm.predict(dtest)

# Make Submission
result = pd.DataFrame({"Id": test["Id"], 'Sales': np.expm1(test_probs)})
result.to_csv("../sub/submission%d%d%d.csv"%(dt.day, dt.hour, dt.minute),index=False)


# In[ ]:




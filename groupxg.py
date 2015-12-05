import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import operator
import matplotlib
matplotlib.use("Agg") #Needed to save figures
import matplotlib.pyplot as plt

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)

monthdic = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

def promo2_indicator(row):
    if row['PromoInterval'] is np.NaN:
        return 0
    elif monthdic[row['month']-1] in row['PromoInterval']:
        return 1
    else:
        return 0

# Gather some features
def build_features(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1   
    # Use some properties directly
    features.extend(['CompetitionDistance', 
                      'Promo','Promo2SinceWeek','Promo2SinceYear'])

    # add some more with a bit of preprocessing
    features.append('SchoolHoliday')
    data['SchoolHoliday'] = data['SchoolHoliday'].astype(float)

    features.extend(['StoreType', 'Assortment', 'StateHoliday'])
    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    data.StoreType.replace(mappings, inplace=True)
    data.Assortment.replace(mappings, inplace=True)
    data.StateHoliday.replace(mappings, inplace=True)

    features.extend(['month', 'day', 'year'])
    temp = data[data.CompetitionOpenSinceMonth.notnull()]
    data['CompetitionTime'] = temp.month - temp.CompetitionOpenSinceMonth + (temp.year - temp.CompetitionOpenSinceYear) * 12
    temp = data[data.Promo2SinceWeek.notnull()]
    data['Promo2TimeInterval'] = 52 - temp.Promo2SinceWeek + (temp.year - temp.Promo2SinceYear) * 52
    features.extend(['Promo2Indicator', 'CompetitionTime','Promo2TimeInterval'])
    #Create Weekday variable
    data['Weekday'] = data.Date.dt.weekday
    data['Quarter'] = data.Date.dt.quarter
    data['WeekOfYear'] = data.Date.dt.week
    features.extend(['Weekday','Quarter','WeekOfYear'])


## Start of main script

print("Load the training, test and store data using pandas")
types = {'CompetitionOpenSinceYear': np.dtype(int),
         'CompetitionOpenSinceMonth': np.dtype(int),
         'StateHoliday': np.dtype(str),
         'Promo2SinceWeek': np.dtype(int),
         'SchoolHoliday': np.dtype(float),
         'PromoInterval': np.dtype(str),
         'CompetitionDistance':np.dtype(float),
         'StoreType':np.dtype(str),
         'Assortment':np.dtype(str)}

train = pd.read_csv("train_feature.csv",parse_dates=[2])
test = pd.read_csv("test_feature.csv",parse_dates=[3])

features = []

print("augment features")
print("build train features")
build_features(features, train)
print("build test features")
build_features([], test)
print(features)

print('training data processed')

params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.005,
          "max_depth": 4,
          "subsample": 0.8,
          "colsample_bytree": 0.7,
          "silent": 1,
          "seed": 1
          }
num_boost_round = 5000

grouped_train = train.groupby('Store')
models = {}
print("Train a XGBoost model")
i = 0
for name, group in grouped_train:
    print "The i-th group", i
    X_train, X_valid = train_test_split(group, test_size=0.01,random_state=10)
    y_train = np.log1p(X_train.Sales)
    y_valid = np.log1p(X_valid.Sales)
    dtrain = xgb.DMatrix(X_train[features],y_train)
    dvalid = xgb.DMatrix(X_valid[features],y_valid) 
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=True)
    print("Validating the i-th group")
    yhat = gbm.predict(xgb.DMatrix(X_valid[features]))
    error = rmspe(X_valid.Sales.values, np.expm1(yhat))
    print('RMSPE for i-the group: {:.6f}'.format(error))
    models[name] = gbm
    i += 1

print("Make predictions on the test set")
grouped_test = test.groupby('Store')
i = 0
for name, group in grouped_test:
    dtest = xgb.DMatrix(group[features])
    test_predict = models[name].predict(dtest)
    if i == 0:
         result = pd.DataFrame({"Id": group["Id"], 'Sales': np.expm1(test_predict)})
    else:
         result = result.append(pd.DataFrame({"Id": group["Id"], 'Sales': np.expm1(test_predict)}))
    i += 1

# Make Submission
result.to_csv("groupxg.csv", index=False)



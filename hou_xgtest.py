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
    #features.extend(['Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth','CompetitionOpenSinceYear', 'Promo', 'Promo2', 'Promo2SinceWeek','Promo2SinceYear'])
    features.extend(['Store', 'CompetitionDistance', 
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
    #data['year'] = data.Date.dt.year
    #data['month'] = data.Date.dt.month
    #data['day'] = data.Date.dt.day
    #data['DayOfWeek'] = data.Date.dt.dayofweek
    #data['Promo2Indicator'] = data.apply(promo2_indicator, axis = 1)
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
#train = pd.read_csv("train.csv", parse_dates=[2], dtype=types)
#test = pd.read_csv("test.csv", parse_dates=[3], dtype=types)
#store = pd.read_csv("store_new.csv")
train = pd.read_csv("train_feature.csv",parse_dates=[2])
test = pd.read_csv("test_feature.csv",parse_dates=[3])

#print("Assume store open, if not provided")
#train.fillna(1, inplace=True)
#test.fillna(1, inplace=True)

#print("Consider only open stores for training. Closed stores wont count into the score.")
#train = train[train["Open"] != 0]
#print("Use only Sales bigger than zero. Simplifies calculation of rmspe")
#train = train[train["Sales"] > 0]

#print("Join with store")
#train = pd.merge(train, store, on='Store')
#test = pd.merge(test, store, on='Store')
#print train[train.PromoInterval.notnull()].head()

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
          "eta": 0.05,
          "max_depth": 10,
          "subsample": 0.9,
          "colsample_bytree": 0.7,
          "silent": 1,
          "seed": 1301
          }
num_boost_round = 7000

print("Train a XGBoost model")
X_train, X_valid = train_test_split(train, test_size=0.012, random_state=10)
y_train = np.log1p(X_train.Sales)
y_valid = np.log1p(X_valid.Sales)
dtrain = xgb.DMatrix(X_train[features], y_train)
dvalid = xgb.DMatrix(X_valid[features], y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
  early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=True)

print("Validating")
yhat = gbm.predict(xgb.DMatrix(X_valid[features]))
error = rmspe(X_valid.Sales.values, np.expm1(yhat))
print('RMSPE: {:.6f}'.format(error))

#print("Make predictions on the traini#ng set")
#train_d = xgb.DMatrix(train[features])
#train_probs = gbm.predict(train_d)
#train_result = pd.DataFrame({"Store":train["Store"],"Date":train["Date"],"Sales":np.expm1(train_probs),"Residuals":train.Sales.values-np.expm1(train_probs)})
#train_result.to_csv("train_prediction.csv",index=False)

print("Make predictions on the test set")
dtest = xgb.DMatrix(test[features])
test_probs = gbm.predict(dtest)
# Make Submission
result = pd.DataFrame({"Id": test["Id"], 'Sales': np.expm1(test_probs)})
result.to_csv("xgboost_1_submission.csv", index=False)

# XGB feature importances
# Based on https://www.kaggle.com/mmueller/liberty-mutual-group-property-inspection-prediction/xgb-feature-importance-python/code

create_feature_map(features)
importance = gbm.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance.png', bbox_inches='tight', pad_inches=1)


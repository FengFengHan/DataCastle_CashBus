import pandas as pd
import numpy as np
import time
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from hyperopt import fmin, hp, STATUS_OK, Trials, tpe
import os
import csv
import datetime
import matplotlib.pyplot as plt

#from myUtil import *

i_path = '/Users/HAN/Documents/CashBus/input/'
o_path = '/Users/HAN/Documents/CashBus/output/'
test_path = '/Users/HAN/Documents/CashBus/test/'
def catToDummy(data, featuresType):
    for i in range(len(featuresType.ix[:,1])):
        if featuresType.ix[i, 1] == 'category':
            feature = featuresType.ix[i,0]
            if feature in data.columns:
                dummies = pd.get_dummies(data[feature], prefix= feature)
                # feature replaced by dummies
                data.drop(feature, axis = 1, inplace = True)
                # for regression, dummies remove last col   ### ???
                ### data = pd.concat([data, dummies.ix[:,:-1]], axis = 1)
                data = pd.concat([data, dummies], axis = 1)
    return data

#def getMissrate(data):
#   Missrate = {}
#   Count = len(data)
#   for feature in data.columns:
#       missCount = len(data[np.isnan(data[feature])])
#       rate = missCount / Count
#       Missrate[feature] = rate
#   return Missrate
def getMissrate(x):
     missCount = np.isnan(x).sum()
     return missCount/len(x)

#read
train_x = pd.read_csv(i_path + 'train_x.csv') # 15000 rows
train_y = pd.read_csv(i_path + 'train_y.csv')
test_x = pd.read_csv(i_path + 'test_x.csv') # 5000 rows
featuresType = pd.read_csv(i_path + 'features_type.csv')
uidType = pd.DataFrame({'feature':['uid'],
                   'type':['numeric']})
featuresType = pd.concat([featuresType, uidType], ignore_index=True
                         )
#miss value
# ##negtive that is less than -2, but it is not missing value
# ### 67 rows,both numeric feature; {feature:rowCount}--
#        --{x329:1, x357:21, x398:2, x950:22, x952:64, x953:6}
# train_x[np.any(train_x  < -2, axis = 1)]
# ### 15 rows, both numeric feature; {x357:1, x950:3, x952:15, x953:1}
# tmp = test_x[np.any(test_x < -2, axis = 1)]
# negFeatures = {}
# for row in range(len(tmp)):
#     for feature in range(len(tmp.columns)):
#         if tmp.iloc[row, feature] < -2:
#             if feature not in negFeatures:
#                 negFeatures[feature] = 0
#             negFeatures[feature] += 1
#             print(tmp.iloc[row, feature])

# missvalue to NA;assign NA  to -1 and -2
def notNA(x):
    if x == -1 or x == -2:
        return False
    return True
def isNA(x):
    if x == -1 or x == -2:
        return True
    return False
train_x.where(train_x.applymap(notNA), np.nan, inplace = True)
test_x.where(test_x.applymap(notNA), np.nan, inplace = True)

# #feature Info
#  #feaNum: include 'uid', 1046
# feaNumCnt = len(featuresType[featuresType['type'] == 'numeric'])
# #feaCat: 93
# feaCateCnt = len(featuresType[featuresType['type'] == 'category'])

# missrate
# line missrate:
#train_x_NA = train_x[np.isnan(train_x), axis = 1)] # 15000 rows
#test_x_NA = test_x[np.isnan(test_x), axis = 1)] # 5000 rows
# feature missrate:
#trainMissrate = getMissrate(train_x) #max:0.9678
#testMissrate = getMissrate(test_x) #max: 0.9648

# trFeatureMiss = train_x.apply(getMissrate, reduce = False).to_frame()
# trFeatureMiss.columns = ['trMissrate']
# trFeatureMiss = pd.merge(featuresType, trFeatureMiss, left_on = 'feature',
#                          right_index= True)
# teFeatureMiss = test_x.apply(getMissrate, reduce = False).to_frame()
# teFeatureMiss.columns = ['teMissrate']
# teFeatureMiss = pd.merge(featuresType, teFeatureMiss,  left_on='feature',
#                          right_index= True)
# FeatureMiss = pd.merge(trFeatureMiss, teFeatureMiss)

## has 642 feture has same missrate 0.0012666  ###???
##trFeatureMissInfo = trFeatureMiss.groupby(by = ['trMissrate']).count()

###    mean: 0.0628; 85% = 0.0547, 0.295=<86% ~97% <= 0.33; max:0.895;
#trNumMissInfo = trFeatureMiss[trFeatureMiss['type'] == 'numeric'].describe(
#percentiles = [0.1, 0.6,0.7,0.8, 0.85, 0.86,0.87,0.97,0.98,0.99])
###      mean:0.249; 40% = 0, 63% = 0.039, 0.172 <= 64%~65% <= 0.191,
###       0.228 <= 66%~76% <= 0.33,77% = 0.531, 0.81<= 78%; max:0.968
#trCatMissInfo = trFeatureMiss[trFeatureMiss['type'] == 'category'].describe(
#percentiles = [0.1, 0.4, 0.5, 0.6,0.61, 0.62, 0.63,0.64,
#               0.65, 0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73, 0.76,
#               0.77,0.78,0.79,0.8, 0.85, 0.9,0.97,0.98,0.99])
###     mean: 0.0618; 85% = 0.0547, 0,286 =< 0.86~0.97 <= 0.326;max:0.897
#teNumMissInfo = teFeatureMiss[teFeatureMiss['type'] == 'numeric'].describe(
#percentiles = [0.1, 0.6, 0.7,0.8, 0.85, 0.86,0.87,0.97,0.98,0.99])
###     mean: 0.248; 40% = 0, 63% = 0.0378, 0.163 <= 64%~65% <= 0.182,
###     0.221 <= 66%~76% <= 0.326,77% = 0.528, 0.81<= 78%; max:0.965
#teCatMissInfo = teFeatureMiss[teFeatureMiss['type'] == 'category'].describe(
#percentiles = [0.1, 0.4, 0.5, 0.6,0.61, 0.62, 0.63,0.64,
#               0.65, 0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73, 0.76,
#               0.77,0.78,0.79,0.8, 0.85, 0.9,0.97,0.98,0.99])
# FeatureMiss = FeatureMiss[['feature', 'type', 'trMissrate', 'teMissrate']]
# for row in range(len(FeatureMiss)):
#     if FeatureMiss.ix[row,'trMissrate'] >= 0.06 or FeatureMiss.ix[row,'teMissrate'] >= 0.06:
#         feature = FeatureMiss.ix[row, 'feature']
#         train_x.drop(feature, axis = 1, inplace = True)
#         test_x.drop(feature, axis = 1, inplace = True)

# As xgboost is treat every variable as numeric and support for miss
#fill NA
for feature in train_x.columns[1:]:
    if np.any(featuresType.loc[featuresType['feature'] == feature, 'type'] == 'category'):
        train_x[feature].fillna(-1, inplace = True)
        test_x[feature].fillna(-1, inplace = True)
# train_x.fillna(train_x.mean(), inplace = True)
# test_x.fillna(test_x.mean(),inplace = True)
train = pd.merge(train_x, train_y)


#create dummy variable for regression
sigVal = -100
test_x['y'] = sigVal
datas = pd.concat([train,test_x])
datas = catToDummy(data=datas, featuresType =featuresType)
train = datas[datas['y'] != sigVal]
test = datas[datas['y'] == sigVal]

##negtive class and positive class
#rate = neg / pos
#posCount = len(train_y[train_y['y'] == 1])  1542
#negCount = len(train_y[train_y['y'] == 0])   13458
#rate = negCount/posCount = 0.11

#get param from log
i_log_path = o_path
file_name = 'T0221_1426_Ffeat0_Mxgb_tree_hyper.csv'
log_file = i_log_path + file_name
params_df = pd.read_csv(log_file)
params_df.sort("res_mean", ascending = False, inplace = True)
param = {}
param_key = params_df.columns[4:]
total_y_prob = np.zeros(test.shape[0])
for row in range(5):
    for key in param_key:
        param[key] = params_df.iloc[row][key]
    param['seed'] = np.random.RandomState(2016 + 100*row)
    param['subsample'] = 0.75
    #tarin by selected param
    best_param = param
    dtrain = xgb.DMatrix(data=train.drop(['uid','y'],axis = 1), label=train.y,
                             missing=np.nan)
    bst = xgb.train(best_param,dtrain,num_boost_round=best_param['num_round'])
    dtest = xgb.DMatrix(data=test.drop(['uid','y'],axis = 1), label=test.y,
                        missing=np.nan)
    test_y_prob = bst.predict(dtest,
                              ntree_limit=bst.best_ntree_limit)
    total_y_prob += test_y_prob * 0.2
#output result
result = pd.DataFrame(columns=['uid', 'score'])
result.uid = test.uid
result.score = total_y_prob
fileName = o_path + 'result.csv'
result.to_csv(fileName, index = False)
'''file has someproblem: (1) the col name no "" (2) the last line is empty;
hence there is some process for file
'''
f = open(fileName, 'r')
fileData = f.read()
fileData = fileData.replace('uid,score','"uid","score"')[:-1]
f.close()
cur_time = time.strftime("%m%d_%H%M",time.localtime())
f = open(o_path + cur_time + '_result.csv', 'w')
f.write(fileData)
f.close()


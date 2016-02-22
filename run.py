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

def myAuc(y_true, y_score):
    y_true  = np.array(y_true)
    y_score = np.array(y_score)
    pos_score = y_score[y_true == 1]
    neg_score = y_score[y_true == 0]
    def ufunc_myAuc(negs):
        def my_auc_help(pos):
            return (negs < pos).sum() + 0.5 * ((negs == pos).sum())
        return np.frompyfunc(my_auc_help, 1, 1)
    totalScore = ((ufunc_myAuc(neg_score)(pos_score)).astype(np.float64)).sum()
    count = (pos_score.shape[0])*(neg_score.shape[0])
    auc = totalScore / count
    return auc

def testParam(params,data, times):
    sumScore_val = 0.0
    sumScore_train = 0.0
    for i in range(times):
        train,val = train_test_split(data, test_size=0.25, random_state= i)
        model = LogisticRegression(C = params['C'],class_weight='auto')
        model.fit(X=train.drop(labels=['uid','y'], axis = 1), y = train.y)
        predict_val = model.predict_proba(val.drop(labels=['uid','y'], axis = 1))
        score_val = metrics.roc_auc_score(y_true = val.y, y_score= predict_val[:,1])
        print(score_val)
        # myAuc is approximate equal to metrics.roc_auc_score
        #score_val = myAuc(y_true = val.y, y_score = predict_val[:,1])
        sumScore_val += score_val
        #predict_train = model.predict_proba(train.drop(labels=['uid','y'], axis = 1))
        #score_train = metrics.roc_auc_score(y_true = train.y, y_score= predict_train[:,1])
        # myAuc is approximate equal to metrics.roc_auc_score
        #score_train = myAuc(y_true = train.y, y_score = predict_train[:,1])
        # sumScore_train += score_train
    return (sumScore_val/(times), sumScore_train/(times))
debug = False
n_runs = 3
n_folds = 4
global trial_counter
global log_handler
log_path = o_path + '/Log/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
def hypert_wrapper(param, algo_name):
    time_start = datetime.datetime.now()
    num_train = train.shape[0]
    num_valid = num_train//n_folds
    score = np.zeros((n_runs, n_folds),dtype=float)
    print('------------------------------------')
    global trial_counter
    global log_handler
    trial_counter += 1
    print('trial_counter: %d' %trial_counter)
    print(param)
    for run in range(n_runs):
        print('run: %d' %run)
        rng = np.random.RandomState(2016 + 100 *run)
        indexs = rng.permutation(num_train)
        for fold in range(n_folds):
            time_start_fold = datetime.datetime.now()
            valid_index_start = fold*num_valid
            valid_index_end = (fold +1)*num_valid
            valid_indexs = indexs[valid_index_start:valid_index_end]
            cv_valid = train.iloc[valid_indexs,:]
            train_indexs = np.concatenate((indexs[:valid_index_start],
                                          indexs[valid_index_end:]))
            cv_train = train.iloc[train_indexs,:]
            if algo_name == 'skl_logis':
                model = LogisticRegression(C = param['C'],class_weight='auto')
                model.fit(X=cv_train.drop(labels=['uid','y'], axis = 1), y = cv_train.y)
                pred = model.predict_proba(cv_valid.drop(labels=['uid','y'], axis = 1))
                pred = pred[:,1]
            elif algo_name == 'xgb_tree':
                evals_result = {}
                d_cv_train = xgb.DMatrix(cv_train.drop(labels=['uid','y'],axis = 1),
                                         label =cv_train.y, missing=np.nan)
                d_cv_valid = xgb.DMatrix(cv_valid.drop(labels=['uid','y'],axis = 1),
                                         label =cv_valid.y, missing=np.nan)
                watchlist = [(d_cv_train,'train'),
                             (d_cv_valid, 'valid')]
                #watchlist = [(d_cv_valid, 'valid')]
                bst = xgb.train(param,d_cv_train,num_boost_round=param['num_round'],
                                evals = watchlist, verbose_eval=10,
                                evals_result=evals_result)
                # if debug:
                #     score_train = evals_result['train']['auc']
                #     score_val = evals_result['valid']['auc']
                #     fig = plt.figure(fold)
                #     ax1 = fig.add_subplot(211)
                #     ax1.plot(score_train[50:],'b')
                #     ax2 = fig.add_subplot(212)
                #     ax2.plot(score_val[50:],'r')
                #     save_name = test_path + ('eta_round_%.1f_%run_%d' %(param['eta'],run, fold))
                #     fig.savefig(save_name + '.png')
                #     data_handler = open(save_name + '.csv','w')
                #     data_writer = csv.writer(data_handler)
                #     data_writer.writerow(score_train)
                #     data_writer.writerow(score_val)
                #     data_handler.flush()
                pred = bst.predict(d_cv_valid, ntree_limit=bst.best_ntree_limit)
            else:
                print('the algo %s is wrong' %algo_name)
            score[run][fold] = metrics.roc_auc_score(y_true=cv_valid.y, y_score=pred)
            fold_time = str((datetime.datetime.now() - time_start_fold).seconds)
            print('fold: %d, score: %.6f, time: %ss' %(fold, score[run][fold], fold_time))
            fold += 1
        print('run: %d, score: %.6f' %(run, np.mean(score[run])))
    score_mean = np.mean(score)
    score_std = np.std(score)
    print('mean score: %.6f, std: %.6f' %(score_mean, score_std))
    total_time = str((datetime.datetime.now() - time_start).seconds)
    #write to log
    log_info = [
        '%d' %trial_counter,
        '%.6f' %score_mean,
        '%.6f' %score_std,
        '%s' %total_time
    ]
    for k,v in sorted(param.items()):
        log_info.append('%s' %v)
    writer.writerow(log_info)
    log_handler.flush()
    return {'loss':-score_mean,
             'attachments': {'std': score_std},
            'status': STATUS_OK}

#train and test:
algorithm = 'xgb_tree'
# search optimization param
objective = lambda p:hypert_wrapper(p,algorithm)
param_space = {}
if algorithm == 'skl_logis':
    param_space_skl_logis = {
        'C':hp.quniform('C',0,3,0.1),
        'max_evals':200
    }
    param_space = param_space_skl_logis
elif algorithm == 'xgb_tree':
    #xgboost for Logistic Regression
    xgb_random_seed = 2016
    xgb_max_evals = 200
    params={
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'early_stopping_rounds':10,
    'silent':1,
    'scale_pos_weight': 1542.0/13458.0,
         'eval_metric': 'auc',
    'gamma':0,
    'max_depth': 8,
    'lambda':550,
        'subsample':1.0,
        'colsample_bytree':0.5,
        'min_child_weight':5,
        'eta': 0.3,
    'seed':2016,
    'nthread':8,
    'num_round': 600,
    'max_evals':1
    }
    param_space_xgb_tree = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'early_stopping_rounds':10,
        'scale_pos_weight': 1542.0/13458.0,
        'eta': hp.quniform('eta', 0.2, 0.4, 0.02),
        'gamma':0,
        'lambda':hp.quniform('lambda',300,900,50),
        'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
        'max_depth': hp.quniform('max_depth', 5, 12, 1),
        'subsample': 1,
        'colsample_bytree': hp.quniform('colsample_bytree', 0.2, 0.7, 0.05),
        'num_round': 200,
        'nthread': 8,
        'silent': 1,
        'seed': xgb_random_seed,
        "max_evals": xgb_max_evals
    }
    param_space = param_space_xgb_tree
    if debug:
        param_space = params

feature_name = 'feat0'
model_name = algorithm
log_file = ('%sT%s_F%s_M%s_hyper.csv' %(log_path,
                                        time.strftime("%m%d_%H%M",time.localtime()),
                                        feature_name,
                                        model_name
))
log_handler = open(log_file,'w')
writer = csv.writer(log_handler)
info = ['trail_counter', 'res_mean', 'res_std', 'time']
for k,v in sorted(param_space.items()):
    info.append(k)
writer.writerow(info)
log_handler.flush()
trials = Trials()
trial_counter = 0
best_param = fmin(objective, param_space, algo= tpe.suggest, max_evals= param_space['max_evals'],
                     trials = trials)
print('best_param:')
print(best_param)
for k,v in best_param.items():
    print('%s : %s' %(k,v))
print('trials.trials:')
print(trials.trials)
#retraining on all the data
if algorithm == 'skl_logis':
    model = LogisticRegression(C= best_param['C'], class_weight='auto')
    #remove 'uid'
    model.fit(X= train.drop(labels = ['uid','y'], axis = 1), y = train.y)
    #test
    test_y_prob = model.predict_proba(test.drop(labels=['uid','y'], axis = 1))
    test_y_prob = test_y_prob[:,1]
elif algorithm == 'xgb_tree':
    dtrain = xgb.DMatrix(data=train.drop(['uid','y'],axis = 1), label=train.y,
                         missing=np.nan)
    bst = xgb.train(best_param,dtrain,num_boost_round=best_param['num_round'])
    dtest = xgb.DMatrix(data=test.drop(['uid','y'],axis = 1), label=test.y,
                        missing=np.nan)
    test_y_prob = bst.predict(dtest,
                              ntree_limit=bst.best_ntree_limit)

result = pd.DataFrame(columns=['uid', 'score'])
result.uid = test.uid
result.score = test_y_prob
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




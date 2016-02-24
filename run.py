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

i_path = '/Users/HAN/Documents/CashBus/feat/'
o_path = '/Users/HAN/Documents/CashBus/output/'
test_path = '/Users/HAN/Documents/CashBus/test/'

#input train and test
feat_name = 'feat0'
train_file = i_path + 'train_' + feat_name + '.csv'
train = pd.read_csv(train_file)
test_file = i_path + 'test_' + feat_name + '.csv'
test = pd.read_csv(test_file)

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
    dtest = xgb.DMatrix(data=test.drop(['uid','y'],axis = 1),
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




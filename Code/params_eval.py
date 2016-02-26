import pandas as pd
import numpy as np
import time
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from hyperopt import fmin, hp, STATUS_OK, Trials, tpe
import os
import csv
import datetime
import matplotlib.pyplot as plt

debug = False
n_runs = 3
n_folds = 4
if debug:
    n_runs = 4

data_path = '/Users/HAN/Documents/CashBus/Data/'
feat_path = data_path + 'feat/'
test_path = data_path + 'test/'
log_path = data_path + 'log/original/'
cv_path = data_path + 'cv/'

param_xgb_tree={
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

#param space:
#"skl_logis"
param_space_skl_logis = {
            'C':hp.quniform('C',0,3,0.1),
            'max_evals':200
        }
#"xgb_tree"
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
            'seed': 2016,
            "max_evals": 200
        }
#"lasso
param_space_skl_lasso = {
    'alpha':hp.quniform("alpha",0, 9e-5, 1e-9),
    'max_evals':100
}
#"rige
#max: alpha: 3.346440148495777; mean : 0.67023599999999994
param_space_skl_rige = {
    'alpha': hp.loguniform("alpha", np.log(0.01), np.log(20)),
    'max_evals':200
}
#
param_space_reg_skl_svr = {
    'task': 'reg_skl_svr',
    'C': hp.loguniform("C", np.log(1), np.log(100)),
    'gamma': hp.loguniform("gamma", np.log(0.001), np.log(0.1)),
    'degree': hp.quniform('degree', 1, 5, 1),
    'epsilon': hp.loguniform("epsilon", np.log(0.001), np.log(0.1)),
    'kernel': hp.choice('kernel', ['rbf', 'poly']),
    "max_evals": hyperopt_param["svr_max_evals"],
}

param_spaces = {}
param_spaces['skl_logis'] = param_space_skl_logis
param_spaces['xgb_tree'] = param_space_xgb_tree
param_spaces['skl_lasso'] = param_space_skl_lasso
param_spaces['skl_rige'] = param_space_skl_rige

feat_name = 'feat1'
model = 'skl_lasso'
if debug:
    res_path = test_path
    # xgb_tree debug
    # model = 'xgb_tree'
    # degbug_log_file = data_path + 'log/' +  'T0225_0012_Ffeat1_Mxgb_tree_hyper.csv'
    # debug_params = pd.read_csv(degbug_log_file)
    # param_debug = {}
    # for key in debug_params.columns:
    #     param_debug[key] = debug_params[key][0]
    # param_debug['max_evals'] = 1
    # param_spaces[model] = param_debug
    param_spaces[model] = {
        'alpha': 1.0,
        'max_evals':1
    }
if model in ['skl_rige','skl_lasso']:
    feat_name = feat_name + '_nonan'
print('model: %s, feature: %s' %(model, feat_name))
global trial_counter
global log_handler
global writer

def get_weight(train, balance = True):
    pos_count = np.sum(train.y == 1)
    neg_count = np.sum(train.y == 0)
    num_train = train.shape[0]
    weight = np.zeros(num_train)
    ratio = pos_count/neg_count
    if balance:
        weight[np.array(train.y == 1)] = 1.0
        weight[np.array(train.y == 0)] = ratio
    else:
        weight[np.array(train.y == 1)] = ratio
        weight[np.array(train.y == 0)] = 1.0
    return weight

def hypert_wrapper(param, solution, train):
    time_start = datetime.datetime.now()
    train_weight = get_weight(train,balance=True)
    model = solution['model']
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
            cv_train_x = cv_train.drop(labels=['uid','y'], axis = 1)
            cv_train_label = cv_train.y
            cv_train_weight = train_weight[train_indexs]
            cv_valid_x = cv_valid.drop(labels=['uid','y'], axis = 1)
            if model == 'skl_logis':
                model = LogisticRegression(C = param['C'],class_weight='auto')
                model.fit(X=cv_train_x, y = cv_train_label)
                pred = model.predict_proba(cv_valid_x)
                pred = pred[:,1]
            elif model == 'xgb_tree':
                evals_result = {}
                d_cv_train = xgb.DMatrix(cv_train_x,
                                         cv_train_label, missing=np.nan)
                d_cv_valid = xgb.DMatrix(cv_valid_x,missing=np.nan)
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
            elif model == 'skl_lasso':
                lasso = Lasso(alpha=param["alpha"], normalize=True)
                lasso.fit(cv_train_x,
                          cv_train_label)
                pred = lasso.predict(cv_valid_x)
            elif model == 'skl_rige':
                rige = Ridge(alpha=param["alpha"], normalize=True)
                rige.fit(cv_train_x,
                          cv_train_label)
                pred = rige.predict(cv_valid_x)
            else:
                print('the algo %s is wrong' %model)
            score[run][fold] = metrics.roc_auc_score(y_true=cv_valid.y, y_score=pred)
            fold_time = str((datetime.datetime.now() - time_start_fold).seconds)
            print('fold: %d, score: %.6f, time: %ss' %(fold, score[run][fold], fold_time))
            fold += 1
            cur_save_path = cv_path + ('run%d_fold%d/' %(run,fold))
            if not os.path.exists(cur_save_path):
                os.makedirs(cur_save_path)
            model_name = ('F%s_M%s_%d' %(solution['feat_name'], model, trial_counter))
            file_name = cur_save_path + model_name + ".csv"
            #############
            # if os.path.exists(file_name):
            #     if not debug:
            #         print('%s has exisit' %file_name)
            #         os._exit(1)
            cv_res = pd.DataFrame(columns=['y_pred','y_label'])
            cv_res.y_pred = pred
            cv_res.y_label = cv_valid.y
            cv_res.to_csv(file_name,index=False)
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

def params_evaluation(model,feat_name):
    #input train and test
    train_file = feat_path + 'train_' + feat_name + '.csv'
    train = pd.read_csv(train_file)
    #output result
    log_file = ('%sT%s_F%s_M%s_hyper.csv' %(log_path,
                                            time.strftime("%m%d_%H%M",time.localtime()),
                                            feat_name,
                                            model
    ))
    global log_handler
    log_handler = open(log_file,'w')
    global writer
    writer = csv.writer(log_handler)
    # search optimization param
    solution = {}
    solution['feat_name'] = feat_name
    solution['model'] = model
    objective = lambda p:hypert_wrapper(p,solution,train)
    param_space = param_spaces[model]
    info = ['trail_counter', 'res_mean', 'res_std', 'time']
    for k,v in sorted(param_space.items()):
        info.append(k)
    writer.writerow(info)
    log_handler.flush()
    trials = Trials()
    global trial_counter
    trial_counter = 0
    best_param = fmin(objective, param_space, algo= tpe.suggest, max_evals= param_space['max_evals'],
                         trials = trials)
    print('best_param:')
    print(best_param)
    for k,v in best_param.items():
        print('%s : %s' %(k,v))
    print('trials.trials:')
    print(trials.trials)

params_evaluation(model,feat_name)



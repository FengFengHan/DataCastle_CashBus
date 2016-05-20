import pandas as pd
import numpy as np
import time
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, hp, STATUS_OK, Trials, tpe
import os
import csv
import datetime
import matplotlib.pyplot as plt
from Code.MyUtil import *



#"xgb_linear
# when it is: lambda = 2.5, alpha 应> 0.5
param_space_xgb_linear = {
    'booster': 'gblinear',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta' : 0.003,
    #'lambda' : hp.quniform('lambda', 0, 5, 0.05),
    'lambda':hp.quniform('lambda',10,20,0.5),
    'alpha' : hp.quniform('alpha', 0, 1, 0.1),
    #'lambda_bias' : hp.quniform('lambda_bias', 0, 3, 0.1),
    'num_round': 90,
    'nthread': 8,
    'silent' : 1,
    'seed': 2016,
    "max_evals": 200,
}






#"SVM.SVR
param_space_skl_svr = {
    'task': 'reg_skl_svr',
    'C': hp.loguniform("C", np.log(1), np.log(100)),
    'gamma': hp.loguniform("gamma", np.log(0.001), np.log(0.1)),
    'degree': hp.quniform('degree', 1, 5, 1),
    'epsilon': hp.loguniform("epsilon", np.log(0.001), np.log(0.1)),
    'kernel': hp.choice('kernel', ['rbf', 'poly']),
    "max_evals": 100
}

param_spaces = {}
param_spaces['skl_logis'] = param_space_skl_logis
param_spaces['xgb_tree'] = param_space_xgb_tree
param_spaces['skl_lasso'] = param_space_skl_lasso
param_spaces['skl_svr'] = param_space_skl_svr
param_spaces['xgb_linear'] = param_space_xgb_linear
param_spaces['skl_rige_xgb_tree'] = {'skl_rige':param_space_skl_rige,
                                     'xgb_tree':param_space_xgb_tree}

n_runs = 3 ###???
n_folds = 4

feat_name = 'feat1'
model = 'skl_rige_xgb_tree'
# if model in ['skl_rige','skl_lasso', 'skl_svr']:
#     feat_name = feat_name + '_nonan'


#input train and test
train_file = feat_path + 'train_' + feat_name + '.csv'
train = pd.read_csv(train_file)
cat_feat_names = list(train.columns[1047:])
# train_num = train.drop(labels = cat_feat_names, axis = 1)
# train_cat = train[['uid'] + cat_feat_names + ['y']]
# it is debug: # if no debug will delete it
# if debug:
#     cv_path = test_path
#     model = 'skl_rige'
#     train = train_cat.iloc[100:7400,:]
    #param_spaces[model]['alpha'] = 3.346440148495777

#print('model: %s, feature: %s' %(model, feat_name))
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

def skl_rige_xgb_tree(param_mix, train,test):
    skl_rige_param = param_mix['skl_rige']
    xgb_tree_param = param_mix['xgb_tree']
    #divide data
    part = 0.5
    num_train = train.shape[0]
    part_num = int(num_train * part)
    train_1 = (train[:part_num])
    train_2 = train[part_num:]
    #build rige model
    train_1_cat = train_1[['uid'] + cat_feat_names + ['y']]
    rige = Ridge(alpha=skl_rige_param['alpha'],
                     normalize=True)
    train_1_cat_x = train_1_cat.drop(labels=['uid','y'],axis = 1)
    train_1_cat_label = train_1_cat.y
    rige.fit(train_1_cat_x,train_1_cat_label)
    # rige model stackded xgb_tree
    train_2_num = train_2.drop(labels=cat_feat_names, axis = 1)
    cat_pred = rige.predict(train_2[cat_feat_names])
    train_2_num['cat_pred'] = cat_pred
    test_num = test.drop(labels=cat_feat_names, axis = 1)
    test_cat_pred = rige.predict(test[cat_feat_names])
    test_num['cat_pred'] = test_cat_pred
    for (col1, col2) in zip(list(test_num.columns), list(train_2_num.columns)):
        if col1 != col2:
            print('col1: %s, col2: %s' %(col1, col2))
    evals_result = {}
    train_2_num_x = train_2_num.drop(labels = ['uid','y'],axis =1)
    d_train = xgb.DMatrix(data=train_2_num_x,label=train_2_num.y, missing=np.nan)
    test_num_x = test_num.drop(labels = ['uid','y'],axis = 1)
    d_test = xgb.DMatrix(data=test_num_x,label=test_num.y, missing=np.nan)
    #watchlist = [(d_cv_valid, 'valid')]
    bst = xgb.train(xgb_tree_param,d_train,num_boost_round=xgb_tree_param['num_round'],
                    verbose_eval=10,evals_result=evals_result)
    pred = bst.predict(data=d_test)
    
    return pred

def hypert_wrapper(param, solution, train):
    print(param)
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
            cv_valid_label = cv_valid.y
            if model == 'skl_logis':
                model = LogisticRegression(C = param['C'],class_weight='auto')
                model.fit(X=cv_train_x, y = cv_train_label)
                pred = model.predict_proba(cv_valid_x)
                pred = pred[:,1]

            elif model == 'skl_lasso':
                lasso = Lasso(alpha=param["alpha"], normalize=True)
                lasso.fit(cv_train_x,
                          cv_train_label)
                pred = lasso.predict(cv_valid_x)

            elif model == 'skl_svr':
                scaler = StandardScaler()
                cv_train_x = scaler.fit_transform(cv_train_x)
                cv_valid_x = scaler.transform(cv_valid_x)
                svr = SVR(C=param['C'], gamma=param['gamma'], epsilon=param['epsilon'],
                                        degree=param['degree'], kernel=param['kernel'],
                                        max_iter = 1000)
                svr.fit(cv_train_x, cv_train_label)
                pred = svr.predict(cv_valid_x)
            elif model == 'xgb_linear':
                evals_result = {}
                d_cv_train = xgb.DMatrix(data = cv_train_x,
                                         label = cv_train_label, missing=np.nan)
                d_cv_valid = xgb.DMatrix(data= cv_valid_x,
                                         label=cv_valid_label, missing=np.nan)
                watchlist = [(d_cv_train,'train'),
                             (d_cv_valid, 'valid')]
                #watchlist = [(d_cv_valid, 'valid')]
                bst = xgb.train(param,d_cv_train,num_boost_round= int(param['num_round']),
                                evals = watchlist, verbose_eval=False,
                                evals_result=evals_result)
                # if debug:
                #     score_train = evals_result['train']['auc']
                #     score_val = evals_result['valid']['auc']
                #     fig = plt.figure(trial_counter)
                #     ax1 = fig.add_subplot(211)
                #     ax1.plot(score_train[50:],'b')
                #     ax2 = fig.add_subplot(212)
                #     ax2.plot(score_val[50:],'r')
                #     save_name = test_path + ('e%.4f_r%d' %(param['eta'], param['num_round'],
                #                                                      ))
                #     fig.savefig(save_name + '.png')
                #     data_handler = open(save_name + '.csv','w')
                #     data_writer = csv.writer(data_handler)
                #     data_writer.writerow(score_train)
                #     data_writer.writerow(score_val)
                #     data_handler.flush()
                pred = bst.predict(d_cv_valid)
            elif model == 'skl_rige_xgb_tree':
                pred = skl_rige_xgb_tree(param, cv_train, cv_valid)
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
            ori_file_name = data_path + ('cv/run%d_fold%d/' %(run,fold))\
                            + model_name + ".csv"
           #############
#            if model in ['xgb_tree', 'skl_rige', 'skl_lasso']:
#                if not os.path.exists(ori_file_name):
#                   print('it is wrong: %s not exists' %ori_file_name)
#                   os._exit(1)
#                if file_name == ori_file_name:
#                   print('%s has exisit' %file_name)
#                   os._exit(1)
            cv_res = pd.DataFrame(columns=['y_pred','y_label'])
            cv_res.y_pred = pred
            cv_res.y_label = cv_valid.y
            cv_res.to_csv(file_name,index=False)
        print('run: %d, score: %.6f' %(run, np.mean(score[run])))
        
        #np.save(bug_file, score)
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

def params_evaluation(model,feat_name,train):
    
    #output result
    log_file = ('%sT%s_F%s_M%s_hyper.csv' %(log_write_path,
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
            
    best_param = fmin(objective, param_space, algo= tpe.suggest, max_evals= 100,
                        trials = trials)
    print('best_param:')
    print(best_param)
    for k,v in best_param.items():
        print('%s : %s' %(k,v))
    print('trials.trials:')
    print(trials.trials)

params_evaluation(model,feat_name,train)

# param = {'skl_rige': {'max_evals': 200, 'alpha': 2.33},
#          'xgb_tree': {'subsample': 1, 'max_depth': 11.0, 'objective': 'binary:logistic', 'num_round': 200, 'colsample_bytree': 0.25, 'silent': 1, 'eval_metric': 'auc', 'min_child_weight': 2.0, 'max_evals': 200, 'early_stopping_rounds': 10, 'seed': 2016, 'scale_pos_weight': 0.11457868925546144, 'eta': 0.4, 'gamma': 0, 'booster': 'gbtree', 'lambda': 550.0, 'nthread': 8}}
# test_file = feat_path + 'test_' + feat_name + '.csv'
# test= pd.read_csv(test_file)
# test["y"] = np.zeros(5000)
# result = pd.DataFrame(columns=['uid', 'score'])
# result.uid = test.uid
# test_pred = skl_rige_xgb_tree(param,train,test)
# result.score = test_pred
# res_path = data_path + 'result/'
# fileName = res_path + 'result.csv'
# result.to_csv(fileName, index = False)
# '''file has someproblem: (1) the col name no "" (2) the last line is empty;
# hence there is some process for file
# '''
# f = open(fileName, 'r')
# fileData = f.read()
# fileData = fileData.replace('uid,score','"uid","score"')[:-1]
# f.close()
# cur_time = time.strftime("%m%d_%H%M",time.localtime())
# f = open(res_path + cur_time + '_result.csv', 'w')
# f.write(fileData)
# f.close()








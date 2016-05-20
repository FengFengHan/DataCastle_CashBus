import matplotlib.pyplot as plt
from hyperopt import fmin, hp, STATUS_OK, Trials, tpe
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import scale
import xgboost as xgb
from Code.MyUtil import *
import pandas as pd
import os
import csv
import datetime
import time

#feat is NA
feat_na = {'xgb_tree':""}
global trial_counter
global log_handler
global log_writer

'''
2016.4.24 15:23, {n_runs: 3, n_folds:4} to {n_runs:1, n_folds:9}
'''

class MyModel:
    param_space = None
    def __init__(self,param=None):
        self.param_ = param

    def run(self,train_x,train_y,test_x,param = None):
        if param is None:
            param = self.param_
        pred = None
        return pred

    def cv_run(self,train_x,train_y,n_runs,n_folds,param = None):
        num_train = len(train_x)
        num_valid = num_train / (n_folds * 1.0)
        scores = np.zeros((n_runs, n_folds), dtype=float)
        for run_count in range(n_runs):
            print('run: %d' % run_count)
            rng = np.random.RandomState(2016 + 100 * run_count)
            indexs = rng.permutation(num_train)
            for fold_count in range(n_folds):
                start_fold_time = datetime.datetime.now()
                valid_index_start = fold_count * num_valid
                valid_index_end = (fold_count + 1) * num_valid
                valid_indexs = indexs[valid_index_start:valid_index_end]
                train_indexs = np.concatenate((indexs[:valid_index_start],
                                               indexs[valid_index_end:]))
                cv_train_x = train_x.iloc[train_indexs,:]
                cv_train_y = train_y[train_indexs]
                cv_valid_x = train_x.iloc[valid_indexs,:]
                cv_valid_y = train_y[valid_indexs]
                pred = self.run(cv_train_x,cv_train_y,cv_valid_x,param)
                score = metrics.roc_auc_score(y_true=cv_valid_y,y_score=pred)
                scores[run_count][fold_count] = score
                fold_time = str((datetime.datetime.now() - start_fold_time).seconds)
                print('fold: %d, score: %.6f, time: %ss' % (fold_count, score, fold_time))
            print('run: %d, score: %.6f' % (run_count, np.mean(scores[run_count])))
        # np.save(bug_file, score)
        score_mean = np.mean(scores)
        score_std = np.std(scores)
        print('mean score: %.6f, std: %.6f' % (score_mean, score_std))
        return (score_mean,score_std)

    def bag_run(self,train_x,train_y,test_x, bag_rate,bag_times,param = None):
        train_num = len(train_x)
        bag_num =int(train_num*bag_rate)
        test_num = len(test_x)
        preds = np.zeros((test_num,bag_times))
        for i in range(bag_times):
            rng = np.RandomState(2016 + 100*i)
            indexs = rng.permutation(train_num)
            indexs_cur = indexs[:bag_num]
            train_cur_x = train_x.iloc[indexs_cur,:]
            train_cur_y = train_y[indexs_cur]
            preds[:,i] = self.run(train_cur_x,train_cur_y,test_x,param)
        pred_final = np.mean(preds,axis=1)
        return pred_final

    def param_sel_var(self, var, start, end, step, param, train_x, train_y, fig_name, log_name):
        var_range = []
        while start <= end:
            var_range.append(start)
            start += step
        df = pd.DataFrame(columns=[var,'score_mean','score_std'])
        df[var] = var_range
        for row in range(len(df)):
            param[var] = df.loc[row,var]
            score_mean, score_std = self.cv_run(train_x,train_y,1,9,param)
            df.loc[row,'score_mean'] = score_mean
            df.loc[row,'score_std'] = score_std
        plt.plot(df[var],df['score_mean'],'-*')
        plt.savefig(fig_name)
        df.sort(columns=['score_mean'],ascending = False, inplace = True)
        df.to_csv(log_name,index=False)

    def hyper_wrapper(self,param,train_x,train_y):
        global trial_counter
        print("trial: %d" %trial_counter)
        #print("param")
        start_time = datetime.datetime.now()
        score_mean, score_std = self.cv_run(train_x,train_y,1,9,param)
        total_time = ( datetime.datetime.now() - start_time ).seconds
        # write to log
        log_info = [
            '%d' % trial_counter,
            '%.6f' % score_mean,
            '%.6f' % score_std,
            '%s' % total_time
        ]
        for k, v in sorted(param.items()):
            log_info.append('%s' % v)
        writer.writerow(log_info)
        log_handler.flush()
        trial_counter += 1
        return {'loss': -score_mean,
                'attachments': {'std': score_std},
                'status': STATUS_OK}

    def param_sel_space(self, param_space, train_x, train_y, log_file_name):
        objective = lambda p:self.hyper_wrapper(p,train_x,train_y)
        global log_handler
        log_handler = open(log_file_name, 'w')
        global writer
        writer = csv.writer(log_handler)
        global trial_counter
        trial_counter = 0
        info = ['trail_counter', 'score_mean', 'res_std', 'time']
        for k, v in sorted(param_space.items()):
            info.append(k)
        writer.writerow(info)
        log_handler.flush()
        trials = Trials()
        best_param = fmin(objective, param_space, algo=tpe.suggest, max_evals=100,
                          trials=trials)
        print('best_param:')
        print(best_param)
        for k, v in best_param.items():
            print('%s : %s' % (k, v))
        print('trials.trials:')
        print(trials.trials)

class SklRige(MyModel):
    # feat2: {alpha:  2.8179, res_mean: 0.658571 }
    id_ = 'skl_rige'
    def __init__(self,param = None):
        MyModel.__init__(self,param)
        if param is None:
            self.param_ = {'alpha':3.3464}

    # "rige
    # max: alpha: 3.346440148495777; mean : 0.67023599999999994
    param_space = {
        'alpha': hp.quniform("alpha", 2.6, 2.9,0.0001),
    }

    def run(self,train_x,train_y,test_x,param = None):
        if param is None:
            param = self.param_
        rige = Ridge(alpha = param["alpha"], normalize=True)
        rige.fit(train_x,
                 train_y)
        pred = rige.predict(test_x)
        return pred

class SklLasso(MyModel):
    # feat2: {alpha: 3.82e-5; mean: 0.67453
    id_ = 'skl_lasso'
    def __init__(self,param = None):
        MyModel.__init__(self,param)
        if param is None:
            self.param_ = {'alpha':4.05e-5}

    # "lasso
    # feate1: max: alpha: 4.05e-5; mean: 0.67897099999999999
    param_space_skl_lasso = {
        'alpha': hp.quniform("alpha", 0, 9e-5, 1e-9),
    }

    def run(self,train_x,train_y,test_x,param = None):
        if param is None:
            param = self.param_
        lasso = Lasso(alpha=param["alpha"], normalize=True)
        lasso.fit(train_x,train_y)
        pred = lasso.predict(test_x)
        return pred

class XgbTree(MyModel):
    id_ = 'xgb_tree'
    # "xgb_tree"
    # eta:0.3~0.4
    # lambda:600以上
    # col: 0.4 以下
    # 'min_child_weight': 0~2
    # 'max_depth':7~11
    param_space = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'early_stopping_rounds': 10,
        'scale_pos_weight': 1542.0 / 13458.0,
        'eta': hp.quniform('eta', 0.2, 0.4, 0.02),
        'gamma': 0,
        'lambda': hp.quniform('lambda', 500, 900, 50),
        'min_child_weight': hp.quniform('min_child_weight', 0, 5, 1),
        'max_depth': hp.quniform('max_depth', 5, 12, 1),
        'subsample': 1,
        'colsample_bytree': hp.quniform('colsample_bytree', 0.2, 0.5, 0.05),
        'num_round': 200,
        'silent': 1,
        'seed': 2016,
        "max_evals": 200
    }

    def __init__(self,param=None):
        MyModel.__init__(self,param)
        if param is None:
            self.param_ = {'booster':'gbtree',
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

    def run(self,train_x,train_y,test_x,param = None):
        if param is None:
            param = self.param_
        dtrain = xgb.DMatrix(train_x,
                                 train_y, missing=np.nan)
        dtest = xgb.DMatrix(test_x,
                                 missing=np.nan)
        bst = xgb.train(param, dtrain, num_boost_round=param['num_round'])
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
        pred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
        return pred

    def param_similar(self, param_other,threshold_):
        same_num = 0
        keys = ['eta', 'lambda', 'min_child_weight',
                'max_depth', 'colsample_bytree']
        for key in keys:
            if self.param_[key] == param_other[key]:
                same_num += 1
        if (same_num >= threshold_):
            return True
        return False

class SklLrl1(MyModel):
    # feat_2: 'C': 0.0208, 'class_weight': 'auto', mean: 0.676348986523
    id_ = 'skl_lrl1'
    def __init__(self,param = None):
        MyModel.__init__(self,param)
        if param is None:
            self.param_ = {'C':1.0, 'class_weight':'auto'}

    def run(self,train_x,train_y,test_x,param = None):
        if param is None:
             param = self.param_
        model = LogisticRegression(penalty='l1', C=param['C'], class_weight=param.get('class_weight','auto'))
        train_x = scale(train_x)
        test_x = scale(test_x)
        model.fit(X=train_x, y= train_y)
        pred = model.predict_proba(test_x)
        pred = pred[:, 1]
        return pred

class SklLrl2(MyModel):
    # feat_2: 'C': 2e-4 + 3e-5, 'class_weight': 'auto', mean: ~0.65734
    id_ = 'skl_lrl2'
    def __init__(self, param=None):
        MyModel.__init__(self,param)
        if param is None:
            self.param_ = {'C': 1.0, 'class_weight': 'auto'}

    def run(self, train_x, train_y, test_x, param=None):
        if param is None:
            param = self.param_
        model = LogisticRegression(C=param['C'], class_weight=param.get('class_weight','auto'),solver = 'newton-cg')
        train_x = scale(train_x)
        test_x = scale(test_x)
        model.fit(X=train_x, y=train_y)
        pred = model.predict_proba(test_x)
        pred = pred[:, 1]
        return pred









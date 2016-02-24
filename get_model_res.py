import pandas as pd
import numpy as np
import datetime
import xgboost as xgb
from sklearn import metrics
import os

i_path = '/Users/HAN/Documents/CashBus/feat/'
o_path = '/Users/HAN/Documents/CashBus/output/'
test_path = '/Users/HAN/Documents/CashBus/test/'
save_cv_path = '/Users/HAN/Documents/CashBus/output/cv/'
save_pred_path = '/Users/HAN/Documents/CashBus/output/pred/'
#input train and test
feat_name = 'feat0'
train_file = i_path + 'train_' + feat_name + '.csv'
train = pd.read_csv(train_file)
test_file = i_path + 'test_' + feat_name + '.csv'
test = pd.read_csv(test_file)

#cv test
n_runs = 3
n_folds = 4
def model_cv_res(param, model_name):
    time_start = datetime.datetime.now()
    num_train = train.shape[0]
    num_valid = num_train//n_folds
    score = np.zeros((n_runs, n_folds),dtype=float)
    print('------------------------------------')
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

            #evals_result = {}
            d_cv_train = xgb.DMatrix(cv_train.drop(labels=['uid','y'],axis = 1),
                                     label =cv_train.y, missing=np.nan)
            d_cv_valid = xgb.DMatrix(cv_valid.drop(labels=['uid','y'],axis = 1),
                                     label =cv_valid.y, missing=np.nan)
            watchlist = [(d_cv_valid, 'valid')]
            #watchlist = [(d_cv_valid, 'valid')]
            bst = xgb.train(param,d_cv_train,num_boost_round=param['num_round'],
                            evals = watchlist, verbose_eval=False)
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
            score[run][fold] = metrics.roc_auc_score(y_true=cv_valid.y, y_score=pred)
            fold_time = str((datetime.datetime.now() - time_start_fold).seconds)
            print('fold: %d, score: %.6f, time: %ss' %(fold, score[run][fold], fold_time))
            fold += 1
            cur_save_path = save_cv_path + ('run%d_fold%d/' %(run,fold))
            if not os.path.exists(cur_save_path):
                os.makedirs(cur_save_path)
            file_name = cur_save_path + model_name + ".csv"
            cv_res = pd.DataFrame(columns=['y_pred','y_label'])
            cv_res.y_pred = pred
            cv_res.y_label = cv_valid.y
            cv_res.to_csv(file_name,index=False)
        print('run: %d, score: %.6f' %(run, np.mean(score[run])))
    score_mean = np.mean(score)
    score_std = np.std(score)
    print('mean score: %.6f, std: %.6f' %(score_mean, score_std))

def model_res(model):
    param = model['param']
    model_name = model['model_name']
    #trainning on All data
    bag_size = 10
    bag_ratio = 0.75
    num_train = train.shape[0]
    num_test = test.shape[0]
    preds_bag = np.zeros((num_test, bag_size))
    for bag_iter in range(bag_size):
        rng = np.random.RandomState(2016 + 100*bag_iter)
        randnums = rng.uniform(size = num_train)
        index_sels = [i for i in range(num_train) if randnums[i] <= bag_ratio]
        train_bag = train.iloc[index_sels,:]
        dtrain = xgb.DMatrix(train_bag.drop(labels=['uid','y'],axis = 1),
                                         label =train_bag.y, missing=np.nan)
        bst = xgb.train(param,dtrain,num_boost_round=param['num_round'])
        dtest = xgb.DMatrix(test.drop(labels=['uid','y'], axis = 1),
                            missing= np.nan)
        pred = bst.predict(dtest, ntree_limit= bst.best_ntree_limit)
        preds_bag[:,bag_iter] = pred
    file_name = save_pred_path + model_name + ".csv"
    res = pd.DataFrame(columns=['pred'])
    res.pred = np.mean(preds_bag,axis=1)
    res.to_csv(file_name,index=False)

#get param from log
i_log_path = o_path
model_id = 'Ffeat0_Mxgb_tree'
file_name = 'T0221_1426_'+ model_id + '_hyper.csv'
log_file = i_log_path + file_name
params_df = pd.read_csv(log_file)
params_df.sort("res_mean", ascending = False, inplace = True)

param = {}
param_key = params_df.columns[4:]
total_y_prob = np.zeros(test.shape[0])


model_num = 10
param_vars = ['eta', 'lambda', 'min_child_weight', 'max_depth', 'colsample_bytree']
threshold = 0.4
model_ranks = []
redundant = False
def model_similar(model1, model2, params, threshold):
    same_num = 0.0
    for param in params:
        if model1[param] == model2[param]:
            same_num += 1
    if (same_num / len(params)) >= threshold:
        return True
    return False
for row in range(params_df.shape[0]):
    redundant = False
    cur_param = params_df.iloc[row,:]
    for model_rank in model_ranks:
        sel_param = params_df.iloc[model_rank,:]
        if model_similar(cur_param,sel_param, param_vars,threshold):
            redundant = True
            break
    if not redundant:
        model_ranks.append(row)
    if len(model_ranks) >= 10:
        break

for model_rank in model_ranks:
    print('model_rank: %d' %model_rank)
    param_all = params_df.iloc[model_rank,:]
    model_counter = param_all['trail_counter']
    print("model: %d, res_mean: %.6f" %(model_counter,
                                      param_all['res_mean']))
    model_name = model_id + ("_%d" %model_counter)
    for key in param_key:
        param[key] = param_all[key]
    model_cv_res(param,model_name)
    # model = {}
    # model['param'] = param
    # model['model_name'] = model_name
    # model_res(model)



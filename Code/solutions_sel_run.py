import pandas as pd
import numpy as np
import datetime
import xgboost as xgb
import os
from sklearn.linear_model import LogisticRegression, Ridge
import csv

data_path = '/Users/HAN/Documents/CashBus/Data/'
feat_path = data_path + 'feat/'
test_path = data_path + 'test/'
log_path = data_path + 'log/'
pred_path = data_path + 'pred/'

debug = False
if debug:
    pred_path = test_path
feat_name = 'feat1'
model = 'xgb_tree'
if model in ['skl_lasso', 'skl_rige']:
    feat_name = feat_name + '_nonan'
solution_name = ('F%s_M%s' %(feat_name, model))
#input train and test
train_file = feat_path + 'train_' + feat_name + '.csv'
train = pd.read_csv(train_file)
test_file = feat_path + 'test_' + feat_name + '.csv'
test = pd.read_csv(test_file)
test_x = test.drop(labels=['uid'], axis = 1)

global file_name
def solution_run(param,solution_id):
    #trainning on All data
    bag_size = 10
    if debug:
        bag_size = 1
    bag_ratio = 0.75
    num_train = train.shape[0]
    num_test = test.shape[0]
    preds_bag = np.zeros((num_test, bag_size))
    for bag_iter in range(bag_size):
        print('bag num: %d' %bag_iter)
        rng = np.random.RandomState(2016 + 100*bag_iter + 10 * solution_id)
        randnums = rng.uniform(size = num_train)
        index_sels = [i for i in range(num_train) if randnums[i] <= bag_ratio]
        train_bag = train.iloc[index_sels,:]
        train_bag_x = train_bag.drop(labels=['uid','y'],axis = 1)
        train_bag_label = train_bag.y
        if model == 'xgb_tree':
            dtrain = xgb.DMatrix(train_bag_x,
                                 label =train_bag_label, missing=np.nan)
            bst = xgb.train(param,dtrain,num_boost_round=param['num_round'])
            dtest = xgb.DMatrix(test_x,
                                missing= np.nan)
            pred = bst.predict(dtest, ntree_limit= bst.best_ntree_limit)
        elif model == 'skl_rige':
            rige = Ridge(alpha=param["alpha"], normalize=True)
            rige.fit(train_bag_x,
                          train_bag_label)
            pred = rige.predict(test_x)
        preds_bag[:,bag_iter] = pred
    global file_name
    file_name = pred_path + solution_name + ('_%d' %solution_id) + ".csv"
    res = pd.DataFrame(columns=['pred'])
    res.pred = np.mean(preds_bag,axis=1)
    res.to_csv(file_name,index=False)


def model_similar(model, param1, param2, thresholds):
    keys = {}
    if model == 'xgb_tree':
        same_num = 0
        keys = ['eta', 'lambda', 'min_child_weight', 
        'max_depth', 'colsample_bytree']
        for key in keys:
            if param1[key] == param2[key]:
                same_num += 1
        if (same_num  >= thresholds[model]):
            return True
    return False

global params_df
global params_df_sel
def solutions_sel(solution_name,log_path):
    #input
    log_file = log_path + solution_name + '_hyper.csv'
    global params_df
    params_df = pd.read_csv(log_file)
    params_df.sort("res_mean", ascending = False, inplace = True)
    # get solution ranks
    max_sel_num = 5
    if debug:
        max_sel_num = 2
    thresholds = {'xgb_tree':3
    }
    solution_ranks = []
    redundant = False
    for row in range(params_df.shape[0]):
        redundant = False
        cur_param = params_df.iloc[row,:]
        for solution_rank in solution_ranks:
            sel_param = params_df.iloc[solution_rank,:]
            if model_similar(model,cur_param,sel_param,thresholds):
                redundant = True
                break
        if not redundant:
            solution_ranks.append(row)
        if len(solution_ranks) >= max_sel_num:
            break

    # get soluntion ids
    # solution_ids = []
    # for solution_rank in solution_ranks:
    #     print('model_rank: %d' %solution_rank)
    #     param_all = params_df.iloc[solution_rank,:]
    #     model_counter = param_all['trail_counter']
    #     print("model: %d, res_mean: %.6f" %(model_counter,
    #                                       param_all['res_mean']))
    #     solution_ids.append(model_counter)

    # get solution sel
    global params_df_sel
    params_df_sel = params_df.iloc[solution_ranks, :]
    #output
    output_file_name = log_path + solution_name + '_sel.csv'
    params_df_sel.to_csv(output_file_name,index=False)

solutions_sel(solution_name, log_path)
solutions_df = pd.read_csv(log_path + solution_name + '_sel.csv')
solutions_df.sort("res_mean", ascending = False, inplace = True)
for row in range(solutions_df.shape[0]):
   print("solution-%d" %row)
   param = (solutions_df.iloc[row, :]).to_dict()
   solution_id = param['trail_counter']
   solution_run(param,solution_id)



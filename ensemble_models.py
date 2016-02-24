import pandas as pd
import numpy as np
import time

#from myUtil import *

i_path = '/Users/HAN/Documents/CashBus/feat/'
o_path = '/Users/HAN/Documents/CashBus/output/'
test_path = '/Users/HAN/Documents/CashBus/test/'

#input train and test
feat_name = 'feat0'
train_file = i_path + 'train_' + feat_name + '.csv'
train = pd.read_csv(train_file)
test_file = i_path + 'test_' + feat_name + '.csv'
test = pd.read_csv(test_file)


#get param from log
i_log_path = o_path
file_name = 'T0221_1426_Ffeat0_Mxgb_tree_hyper.csv'
log_file = i_log_path + file_name
params_df = pd.read_csv(log_file)
params_df.sort("res_mean", ascending = False, inplace = True)
param = {}
param_key = params_df.columns[4:]

pred_path = '/Users/HAN/Documents/CashBus/output/pred/'
model_id = 'Ffeat0_Mxgb_tree'
model_ranks = [0,1,6,9,15,20,25,29,30,32]
model_counters = []
for model_rank in model_ranks:
    param_all = params_df.iloc[model_rank,:]
    model_counters.append(param_all['trail_counter'])

model_num = 10
counters = model_counters[:model_num]
# direct ensemble
num_test = 5000
test_preds = np.zeros((num_test,model_num))
for n in range(len(counters)):
    test_y_prob = pd.read_csv(pred_path + model_id + ('_%d.csv' %counters[n]))
    test_preds[:,n] = np.array(test_y_prob.pred)
test_pred = np.mean(test_preds, axis = 1)


# bag ensemble
# bag_size = 5
# bag_ratio = 0.5
# bag_y_prob = np.zeros(test.shape[0])
# preds = np.zeros((test.shape[0],model_num))
# for model_index in range(model_num):
#     test_y_prob = pd.read_csv(save_path + ('best_model_%d.csv' %model_index))
#     preds[:,model_index] = np.array(test_y_prob.score)
# for bagiter in range(bag_size):
#     rng = np.random.RandomState(2016 + 100 * bagiter)
#     randarray = rng.uniform(size = model_num)
#     bag_indexs = [i for i in range(model_num) if randarray[i] >= bag_ratio]
#     for model_index in bag_indexs:
#         bag_y_prob += preds[:,model_index]
#     bag_y_prob = bag_y_prob * (1.0 / len(bag_indexs))
# bag_y_prob = bag_y_prob * (1.0 / bag_size)

# for row in range(1,10):
#     for key in param_key:
#         param[key] = params_df.iloc[row][key]
#     param['seed'] = np.random.RandomState(2016 + 100*row)
#     param['subsample'] = 0.75
#     #tarin by selected param
#     best_param = param
#     dtrain = xgb.DMatrix(data=train.drop(['uid','y'],axis = 1), label=train.y,
#                              missing=np.nan)
#     bst = xgb.train(best_param,dtrain,num_boost_round=best_param['num_round'])
#     dtest = xgb.DMatrix(data=test.drop(['uid','y'],axis = 1), label=test.y,
#                         missing=np.nan)
#     test_y_prob = bst.predict(dtest,
#                               ntree_limit=bst.best_ntree_limit)
#     file_name = save_path + ('best_model_%d.csv' %row)
#     result = pd.DataFrame(columns=['uid', 'score'])
#     result.uid = test.uid
#     result.score = test_y_prob
#     result.to_csv(file_name, index=False)


#output result
result = pd.DataFrame(columns=['uid', 'score'])
result.uid = test.uid
result.score = test_pred
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


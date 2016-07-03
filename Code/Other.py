import sys
sys.path.extend(['/Users/HAN/AllCode/Projects/CashBus'])
from Code.Models import *
from Code.MyUtil import *
import pandas as pd
from scipy.stats import pearsonr
import numpy as np

train = pd.read_csv(feat_path + 'train_feat3.csv')
train_x = train.drop(labels=['uid','y'],axis=1)
train_y = train['y']
test = pd.read_csv(feat_path + 'test_feat3.csv')
test_x = test.drop(labels=['uid'],axis=1)
test_uids = test['uid']
assert np.all(train_x.columns == test_x.columns)

xgb_res_space = pd.read_csv(log_read_path + 'Ffeat0_Mxgb_tree_hyper.csv')
xgb_res_space.sort_values(by='res_mean',inplace=True,ascending=False)
sel_num = 30
xgb_res_space_sel = xgb_res_space.iloc[:30,:]

xgb_tree = XgbTree()
predicts = pd.DataFrame()
for i in range(len(xgb_res_space_sel)):
    param = (xgb_res_space_sel.iloc[i,:]).to_dict()
    predict = xgb_tree.run(train_x, train_y,test_x,param)
    df = pd.DataFrame(columns=['uid','score'])
    df['uid'] = test_uids
    df['score'] = predict
    file_name = res_path + 'predict_xt_' + str(i) + '.csv'
    df.to_csv(file_name,index=False)
    predicts['score_' + str(i)] = predict
f_mean = lambda x: (x[1:]).mean()
predicts_mean = predicts.apply(f_mean,axis=1)
def f_std(x):
    valid_x = x[1:]
    valid_x_mean = valid_x.mean()
    res = np.sum(np.power((valid_x - valid_x_mean)/valid_x_mean,2))
    return res
predicts_std = predicts.apply(f_std, axis=1)
predicts['scores_mean'] = predicts_mean
predicts['scores_std'] = predicts_std

predict_lrl1 = (SklLrl1()).run(train_x.fillna(train_x.median()),
                               train_y.fillna(train_y.median),
                               test_x.fillna(test_x.median()))

threshold = 0.05
result =pd.DataFrame(columns=['uid','score'])
result['uid'] = test_uids
for i in range(result.shape[0]):
    if predicts.loc[i,'scores_std'] > threshold:
        result.loc[i,'score'] = predict_lrl1.loc[i]
    else:
        result.loc[i,'score'] = predicts.loc[i,'scores_mean']
fileName = res_path + 'result.csv'
result.to_csv(fileName, index=False)
'''file has someproblem: (1) the col name no "" (2) the last line is empty;
hence there is some process for file
'''
f = open(fileName, 'r')
fileData = f.read()
fileData = fileData.replace('uid,score', '"uid","score"')[:-1]
f.close()
cur_time = time.strftime("%m%d_%H%M", time.localtime())
f = open(res_path + cur_time + '_result.csv', 'w')
f.write(fileData)
f.close()

#
# def sel_pearsonr(predicts_test):
#     num = predicts_test.shape[1]
#     selected = {}
#     selected[0] = True
#     sel_count = 1
#     while sel_count < num:
#         sel_count += 1
#         max_mic = 0
#         for i in range(num):
#             if pearsonr(predicts_test,):


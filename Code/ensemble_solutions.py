import pandas as pd
import numpy as np
import time

data_path = '/Users/HAN/Documents/CashBus/Data/'
feat_path = data_path + 'feat/'
test_path = data_path + 'test/'
log_path = data_path + 'log/'
pred_path = data_path + 'pred/'
res_path = data_path + 'result/'
test = pd.read_csv(feat_path + 'test_feat1.csv')

num_test = 5000
solution_names = ['Ffeat1_Mxgb_tree']
preds_num = 0
for solution_name in solution_names:
    params = pd.read_csv(log_path + solution_name + '_sel.csv')
    preds_num += params.shape[0]

test_preds = np.zeros((num_test, preds_num))
pred_count = 0
for solution_name in solution_names:
    ids = []
    params = pd.read_csv(log_path + solution_name + '_sel.csv')
    for solution_id in params.trail_counter:
        ids.append(solution_id)
    for id in ids:
        test_y_prob = pd.read_csv(pred_path + solution_name + ('_%d.csv' %id))
        test_preds[:,pred_count] = np.array(test_y_prob.pred)
        pred_count += 1
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
fileName = res_path + 'result.csv'
result.to_csv(fileName, index = False)
'''file has someproblem: (1) the col name no "" (2) the last line is empty;
hence there is some process for file
'''
f = open(fileName, 'r')
fileData = f.read()
fileData = fileData.replace('uid,score','"uid","score"')[:-1]
f.close()
cur_time = time.strftime("%m%d_%H%M",time.localtime())
f = open(res_path + cur_time + '_result.csv', 'w')
f.write(fileData)
f.close()


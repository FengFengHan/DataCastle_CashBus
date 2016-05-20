import pandas as pd
import numpy as np
import time
from sklearn import metrics
from sklearn.preprocessing import scale
import csv
from Code.MyUtil import *


test = pd.read_csv(feat_path + 'test_feat1.csv')
train = pd.read_csv(feat_path + 'train_feat1.csv')
num_test = test.shape[0]
num_train = train.shape[0]
solution_names = ['Ffeat0_Mxgb_tree',
'Ffeat1_Mxgb_tree', 'Ffeat1_nonan_Mskl_rige','Ffeat1_nonan_Mskl_lasso']


allSolutions = []
for solution_name in solution_names:
    params = pd.read_csv(log_read_path + solution_name + '_sel.csv')
    for solution_id in params.trail_counter:
        solution = solution_name + ('_%d' %solution_id)
        allSolutions.append(solution)
sol_num = len(allSolutions)

n_runs = 3
n_folds = 4
num_valid = 3750
cv_num = n_runs * n_folds
cv_labels = np.zeros((cv_num,num_valid), dtype = int)
cv_preds = np.zeros((cv_num, sol_num, num_valid), dtype= float)

stand = True # cv 的验证证明 stand是有效果的，但提交后变差
# load cv labels
for run in range(n_runs):
    rng = np.random.RandomState(2016 + 100 *run)
    indexs = rng.permutation(num_train)
    for fold in range(n_folds):
        cv_id = run * n_folds + fold
        valid_index_start = fold*num_valid
        valid_index_end = (fold +1)*num_valid
        valid_indexs = indexs[valid_index_start:valid_index_end]
        valid = train.iloc[valid_indexs,:]
        cv_labels[cv_id,:] = valid.y

#load cv preds
scores = np.zeros((n_runs,n_folds))
for i in range(len(allSolutions)):
    solution = allSolutions[i]
    for run in range(n_runs):
        for fold in range(n_folds):
            cv_id = run * n_folds + fold
            fileName = cv_path + ('run%d_fold%d/' %(run, fold+1)) + solution + '.csv'
            cv_res = pd.read_csv(fileName)
            cv_preds[cv_id,i, :] = cv_res.y_pred
            if stand:
                cv_preds[cv_id,i, :] = scale(cv_preds[cv_id,i, :])
            scores[run][fold] = metrics.roc_auc_score(y_true=cv_labels[cv_id,:],y_score=cv_preds[cv_id,i,:])
    scores_mean = np.mean(scores)
    print('%s res_mean: %.6f' %(solution,scores_mean))

#blending
def ensemble_solutions(weights):
    #weights = np.ones((sol_num,1), dtype = float)
    preds_w = np.zeros((sol_num,num_valid), dtype = float)
    pred_blends = np.zeros((n_runs,n_folds,num_valid), dtype= float)
    for run in range(n_runs):
        for fold in range(n_folds):
            cv_id = run * n_folds + fold
            for i in range(len(allSolutions)):
                preds_w[i,:] = cv_preds[cv_id,i,:] * weights[i]
            pred_blends[run,fold,:] = np.sum(preds_w,axis=0) / np.sum(weights)
            scores[run][fold] = metrics.roc_auc_score(y_true=cv_labels[cv_id,:],
                                                      y_score = pred_blends[run][fold])
    scores_mean = np.mean(scores)
    print('weights:')
    print(weights)
    print('blend res mean: %.6f' %scores_mean)
    return scores_mean

# average
ave_res_mean = ensemble_solutions(np.ones((sol_num,1), dtype = float))


#random search
# cur_time = time.strftime("%m%d_%H%M",time.localtime())
# ensemble_res_file = res_path + cur_time + 'ensemble_random.csv'
# handler = open(ensemble_res_file,'w')
# writer = csv.w
#random_num = 10
#best_weights = np.zeros((random_num,sol_num), dtype = float)
#best_scores_means = np.zeros((random_num), dtype=float)
#for random_time in range(random_num):
#    iter_num = 100
#    iter = 0
#    best_scores_mean = 0.0
#    best_weight = np.zeros((sol_num), dtype = float)
#    while iter < iter_num:
#        rng = np.random.RandomState(iter * 100 + 2016 + 7*random_time)
#        weight = rng.uniform(size = sol_num)
#        weight = weight/np.sum(weight)
#        scores_mean = ensemble_solutions(weight)
#        if scores_mean > best_scores_mean:
#            best_scores_mean = scores_mean
#            best_weight = weight
#        iter += 1
#    best_weights[random_time,:] = best_weight
#    best_scores_means[random_time] = best_scores_mean
#grid

#load preds:
#best_weight = np.mean(best_weights,axis=0)
best_weight = np.ones(sol_num, dtype =float)#提交的结果表明均匀提交比较好
test_preds = np.zeros((num_test,sol_num), dtype=float)
for i in range(sol_num):
    solution = allSolutions[i]
    res_pred = pd.read_csv(pred_path + solution + ".csv")
    test_preds[:,i] =  res_pred.pred
    if stand:
        test_preds[:,i] = scale(test_preds[:,i])
    test_preds[:,i] = test_preds[:,i] * best_weight[i]
test_pred = np.sum(test_preds,axis=1) / np.sum(best_weight)
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


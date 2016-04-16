import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

data_path = '/Users/HAN/Documents/CashBus/Data/'
feat_path = data_path + 'feat/'
test_path = data_path + 'test/'
log_path = data_path + 'log/'
pred_path = data_path + 'pred/'
fig_path = data_path + 'fig/'
# # define an objective function
# # def objective(args):
# #     case, val = args
# #     if case == 'case 1':
# #         return val
# #     else:
# #         return val ** 2
# def objective(args):
#     x,y = args
#     return x ** 2 + y ** 2
#
# # define a search space
# from hyperopt import hp
# # space = hp.choice('a',
# #     [
# #         ('case 1', 1 + hp.lognormal('c1', 0, 1)),
# #         ('case 2', hp.uniform('c2', -10, 10))
# #     ])
# space = [hp.uniform('x',0,1), hp.uniform('y',0,1)]
# # minimize the objective over the space
# from hyperopt import fmin, tpe, Trials
# trials = Trials()
# best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials = trials)
#
# print(best)
# # -> {'a': 1, 'c2': 0.01420615366247227}
# import hyperopt
# print(hyperopt.space_eval(space, best))
# # -> ('case 2', 0.01420615366247227}

# import pickle
# import time
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
#
# def objective(x):
#     return {
#         'loss': x ** 2,
#         'status': STATUS_OK,
#         # -- store other results like this
#         'eval_time': time.time(),
#         'other_stuff': {'type': None, 'value': [0, 1, 2]},
#         # -- attachments are handled differently
#         'attachments':
#             {'time_module': pickle.dumps(time.time)}
#         }
# trials = Trials()
# best = fmin(objective,
#     space=hp.uniform('x', -10, 10),
#     algo=tpe.suggest,
#     max_evals=100,
#     trials=trials)
# print(trials.trials)
# #print(best)

# import numpy as np
# from sklearn.cross_validation import KFold
# n_runs = 3
# for run in range(n_runs):
#     rng = np.random.RandomState(2016 + 1000 * run)
#     kf = KFold(n=15000, n_folds=4,random_state=rng)
#     for train_index, valid_index in kf:
#         print('hello %d' %run)

# import pandas as pd
# import csv
# logfile = './temp.csv'
# log_handler = open(logfile,'w')
# col = ['a','b']
# writer = csv.writer(log_handler)
# writer.writerow(col)
# log_handler.flush()
# info = ['1','2']
# writer.writerow(info)
# log_handler.flush()
# content = pd.read_csv(logfile)

# import datetime
# process_start = datetime.datetime.now()
# '''
# process
# '''
# process_end = datetime.datetime.now()
# porcess_time = process_end - process_start
# print('time: %s s' %porcess_time.seconds)

# plot parameter with res_mean
# file = log_path + 'T0228_0934_Ffeat1_Mxgb_linear_hyper.csv'
# params_df = pd.read_csv(file)
# fig = plt.figure(figsize=(14,6))
# # `ax` is a 3D-aware axis instance because of the projection='3d' keyword argument to add_subplot
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# #plt.plot(params_df.alpha, params_df['lambda'],params_df.res_mean,'*')
# tmp = params_df[['alpha', 'lambda', 'res_mean']]
# tmp.to_csv(test_path + 'cur.csv',index=False)

#denote: the scale-pos-weight value in range(0.1~0,2)
# file1 = log_path + 'T0228_2152_Ffeat1_Mxgb_tree_u1_hyper.csv'
# params_df1 = pd.read_csv(file1)
# file2 = log_path + 'T0229_0012_Ffeat1_Mxgb_tree_u1_hyper.csv'
# params_df2 = pd.read_csv(file2)
# params_df = pd.concat([params_df1,params_df2], ignore_index=True)
# plt.plot(params_df['scale_pos_weight'],params_df['res_mean'], '*')
# #plt.show()
# plt.savefig(fig_path + 'xgb_tree-scale_pos-weight.png')

file1 = log_path + 'Ffeat0_Mxgb_tree_hyper.csv'
params_df = pd.read_csv(file1)
#sel_param = params_df[['min_child_weight', 'max_depth', 'res_mean']]
#sel_param.to_csv(test_path + 'mc_md_res.csv', index =False)
#sel_param = params_df[['lambda', 'colsample_bytree', 'res_mean']]
#sel_param.to_csv(test_path + 'ld_cb_res.csv', index =False)
sel_param = params_df[['eta','res_mean']]
sel_param.to_csv(test_path + 'eta_res.csv', index =False)

# X = params_df.alpha
# Y= params_df['lambda']
# Z = params_df.res_mean
# p = ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=0)
# params_df.sort_values(by = 'res_mean', ascending = False,inplace = True)
# max_lambda = (params_df.iloc[0,])['lambda']
# max_alpha = (params_df.iloc[0,])['alpha']
# max_mean = (params_df.iloc[0,])['res_mean']
# plt.show()
#plt.savefig(fig_path + 'xgb_linear_alpha_mean_2.png')

# feat_name = 'feat
# train_file = feat_path + 'train_' + feat_name + '.csv'
# train = pd.read_csv(train_file)
# test_file = feat_path + 'test_' + feat_name + '.csv'
# test = pd.read_csv(test_file)
# test_x = test.drop(labels=['uid'], axis = 1)

# max num valid
# max_num_valid = 0
# def myCrossValidation(data, n_runs, n_folds):
#     num_data = len(data)
#     num_valid = num_data//n_folds
#     for run in range(n_runs):
#         rng = np.random.RandomState(2016 + 100 *run)
#         indexs = rng.permutation(num_data)
#         for fold in range(n_folds):
#             valid_index_start = fold*num_valid
#             valid_index_end = (fold +1)*num_valid
#             valid_indexs = indexs[valid_index_start:valid_index_end]
# #            valid = data.iloc[valid_indexs,:]
# #            train_indexs = np.concatenate((indexs[:valid_index_start],
# #                                          indexs[valid_index_end:]))
# #            train = data.iloc[train_indexs,:]
# #            save_path = save_root + ('run%d_fold%d/' %(run,fold))
# #            if not os.path.exists(save_path):
# #                os.makedirs(save_path)
# #            valid.to_csv(save_path + 'valid.csv',index=False)
# #            train.to_csv(save_path + 'train.csv', index=False)
#             num_valid =len(valid_indexs)
#             print('Valid: %d' %num_valid)
#             global max_num_valid
#             if max_num_valid < num_valid:
#                 max_num_valid = num_valid
# data = [0] * 15000
# myCrossValidation(data,3,4)


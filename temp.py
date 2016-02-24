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
import pandas as pd



import pandas as pd
import numpy as np
import os
save_root = '/Users/HAN/Documents/CashBus/input/cv/'
def myCrossValidation(data, n_runs, n_folds):
    num_data = len(data)
    num_valid = num_data//n_folds
    for run in range(n_runs):
        rng = np.random.RandomState(2016 + 100 *run)
        indexs = rng.permutation(num_data)
        for fold in range(n_folds):
            valid_index_start = fold*num_valid
            valid_index_end = (fold +1)*num_valid
            valid_indexs = indexs[valid_index_start:valid_index_end]
            valid = data.iloc[valid_indexs,:]
            train_indexs = np.concatenate((indexs[:valid_index_start],
                                          indexs[valid_index_end:]))
            train = data.iloc[train_indexs,:]
            save_path = save_root + ('run%d_fold%d/' %(run,fold))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            valid.to_csv(save_path + 'valid.csv',index=False)
            train.to_csv(save_path + 'train.csv', index=False)

# i_path = '/Users/HAN/Documents/CashBus/input/'
# train_x = pd.read_csv(i_path + 'train_x.csv') # 15000 rows
# myCrossValidation(train_x,3,4)




            
            




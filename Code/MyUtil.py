from hyperopt import fmin, hp, STATUS_OK, Trials, tpe
import numpy as np

#path
data_path = '/Users/HAN/AllCode/Projects/CashBus/Data/'
input_path = data_path + 'input/'
output_path = data_path + 'feat/'
feat_path = data_path + 'feat/'
test_path = data_path + 'test/'
log_write_path = data_path + 'log_write/'
log_read_path = data_path + 'log_read/'
cv_path = data_path + 'cv/'
pred_path = data_path + 'pred/'
res_path = data_path + 'result/'
fig_path = data_path + 'fig/'
ensemble_path = data_path + 'ensemble/'
answer_path = data_path + 'answers/'
sol_sel_path = data_path + 'sol_sel'
#function
def get_divides(length_,rate,divide_num):
    '''

    :param length_: length of divided
    :param rate: rate = (length of part)/(length_)
    :param divide_num: number of divide parts
    :return: [ [divides_1],[divides_2],....[divides_(divide_num -1)] ]
    '''
    divides = []
    sel_num = length_*rate
    for i in range(divide_num):
        rng = np.random.RandomState(2016 + 100*i)
        indexs_ = rng.permutation(length_)
        divides.append(indexs_[:sel_num])
    return divides
#solution_sel_path = data_path + 'solution_sel/'
from hyperopt import fmin, hp, STATUS_OK, Trials, tpe
import numpy as np
import copy
import matplotlib.pyplot as plt
import time

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

def cum_plot(values,len_,fig):
  values = copy.copy(values)
  x = sorted(values)
  y = [i for i in range(1,len_+1)]
  plt.plot(x,y)
  plt.savefig(fig)

def submit(result):
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
#solution_sel_path = data_path + 'solution_sel/'
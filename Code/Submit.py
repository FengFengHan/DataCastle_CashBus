from Code.MyUtil import *
from Code.GetEnsWeight import MyAnswer
import time
import pickle
import pandas as pd
from sklearn.preprocessing import scale

answer_file_name = answer_path + 'answer.txt'
answer_file = open(answer_file_name, 'rb')
answers = pickle.load(answer_file)
answer_num = len(answers)

test = pd.read_csv(feat_path + 'test_%s.csv' %(answers[0]).feat_name_)
test_len = test.shape[0]
test_preds = np.zeros((test_len,answer_num))
weights = np.zeros(answer_num)
for i in range(answer_num):
    answer = answers[i]
    #read train and test
    train = pd.read_csv(feat_path + 'train_%s.csv' %answer.feat_name_)
    train_x = train.drop(labels=['uid','y'])
    train_y = train.y
    test = pd.read_csv(feat_path + 'test_%s.csv' %answer.feat_name_)
    test_x = test.drop(labels=['uid'])
    #bag predict
    pred = answer.model_.bag_run(train_x,train_y,test_x, bag_rate = 0.8,bag_times = 10,param = answer.param_)
    test_preds[i] = pred
    weights[i] = answer.weight_

# weight ensemble
for i in range(answer_num):
    test_preds[i] = scale(test_preds[i]) * weights[i]

test_y_prob = np.sum(test_preds,axis=1) / np.sum(weights)

#submit
result = pd.DataFrame(columns=['uid', 'score'])
result.uid = test.uid
result.score = test_y_prob
submit(result)
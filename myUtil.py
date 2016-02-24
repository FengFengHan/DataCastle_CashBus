# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 21:37:54 2016

@author: HAN
"""
#import pandas as pd
# create dummy variable for regression

# from sympy import  *
# y1, y2, lam = symbols('y1 y2 lam')
# f = y1 * log(1+ exp(-y2)) + (1-y1)*log(1 + exp(y2))
# d1 = diff(f, y2)
# d2 = diff(d1, y2)
# ans = simplify(d2 /(d1 + lam) )
#(y1 + (-y1 + (y1 - 1)*exp(y2))*(exp(y2) + 1) - (y1 - 1)*exp(2*y2))/
#((exp(y2) + 1)*(-lam*(exp(y2) + 1) + y1 + (y1 - 1)*exp(y2)))

import pandas as pd
import numpy as np
import time
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold

def myAuc(y_true, y_score):
    y_true  = np.array(y_true)
    y_score = np.array(y_score)
    pos_score = y_score[y_true == 1]
    neg_score = y_score[y_true == 0]
    def ufunc_myAuc(negs):
        def my_auc_help(pos):
            return (negs < pos).sum() + 0.5 * ((negs == pos).sum())
        return np.frompyfunc(my_auc_help, 1, 1)
    totalScore = ((ufunc_myAuc(neg_score)(pos_score)).astype(np.float64)).sum()
    count = (pos_score.shape[0])*(neg_score.shape[0])
    auc = totalScore / count
    return auc

def testParam(params,data, times):
    sumScore_val = 0.0
    sumScore_train = 0.0
    for i in range(times):
        train,val = train_test_split(data, test_size=0.25, random_state= i)
        model = LogisticRegression(C = params['C'],class_weight='auto')
        model.fit(X=train.drop(labels=['uid','y'], axis = 1), y = train.y)
        predict_val = model.predict_proba(val.drop(labels=['uid','y'], axis = 1))
        score_val = metrics.roc_auc_score(y_true = val.y, y_score= predict_val[:,1])
        print(score_val)
        # myAuc is approximate equal to metrics.roc_auc_score
        #score_val = myAuc(y_true = val.y, y_score = predict_val[:,1])
        sumScore_val += score_val
        #predict_train = model.predict_proba(train.drop(labels=['uid','y'], axis = 1))
        #score_train = metrics.roc_auc_score(y_true = train.y, y_score= predict_train[:,1])
        # myAuc is approximate equal to metrics.roc_auc_score
        #score_train = myAuc(y_true = train.y, y_score = predict_train[:,1])
        # sumScore_train += score_train
    return (sumScore_val/(times), sumScore_train/(times))
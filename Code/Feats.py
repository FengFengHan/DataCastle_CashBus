import sys
sys.path.extend(['/Users/HAN/AllCode/Projects/CashBus'])
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
import csv
import matplotlib.pyplot as plt

#from myUtil import *
from Code.MyUtil import *


#debug = True
#if debug:
#    output_path = '/Users/HAN/Documents/CashBus/Data/feat/test/'
#F_name = 'feat0' #2016.2.24 labeled in git
#feat_name = 'feat1' # based on feat0,then 去除categorical中缺失率大于25% ;去除one-hot col of missing
#feat_name = 'feat2' # 去除所有属性中缺失率大于10%的
feat_name = 'feat3' # 不去除有缺失的属性

rem_thresholds = {'feat2': 0.1, 'feat3':1.0}
rem_threshold = rem_thresholds[feat_name]
#def gen_feature(feat_name, input_path,output_path):
def catToDummy(data, featuresType):
    for i in range(len(featuresType.ix[:,1])):
        if featuresType.ix[i, 1] == 'category':
            feature = featuresType.ix[i,0]
            if feature in data.columns:
                dummies = pd.get_dummies(data[feature], prefix= feature)
                # for regression, dummies remove last col   ### ???
                ### data = pd.concat([data, dummies.ix[:,:-1]], axis = 1)
                data = pd.concat([data, dummies], axis = 1)
                # feature replaced by dummies
                data.drop(feature, axis = 1, inplace = True)
    return data

def getMissrate(x):
     missCount = np.isnan(x).sum()
     return missCount/len(x)

#read
train_x = pd.read_csv(input_path + 'train_x.csv') # 15000 rows
train_y = pd.read_csv(input_path + 'train_y.csv')
test_x = pd.read_csv(input_path + 'test_x.csv') # 5000 rows
featuresType = pd.read_csv(input_path + 'features_type.csv')
uidType = pd.DataFrame({'feature':['uid'],
                   'type':['numeric']})
featuresType = pd.concat([featuresType, uidType], ignore_index=True
                         )
# #feature Info
#  #feaNum: include 'uid', 1046
# feaNumCnt = len(featuresType[featuresType['type'] == 'numeric'])
# #feaCat: 93
# feaCateCnt = len(featuresType[featuresType['type'] == 'category'])

##negtive class and positive class
#rate = neg / pos
#posCount = len(train_y[train_y['y'] == 1])  13458
#negCount = len(train_y[train_y['y'] == 0])   1542
#rate = negCount/posCount = 0.11

#miss value
# ##negtive that is less than -2, but it is not missing value
# ### 67 rows,both numeric feature; {feature:rowCount}--
#        --{x329:1, x357:21, x398:2, x950:22, x952:64, x953:6}
# train_x[np.any(train_x  < -2, axis = 1)]
# ### 15 rows, both numeric feature; {x357:1, x950:3, x952:15, x953:1}
# tmp = test_x[np.any(test_x < -2, axis = 1)]
# negFeatures = {}
# for row in range(len(tmp)):
#     for feature in range(len(tmp.columns)):
#         if tmp.iloc[row, feature] < -2:
#             if feature not in negFeatures:
#                 negFeatures[feature] = 0
#             negFeatures[feature] += 1
#             print(tmp.iloc[row, feature])

# missvalue to NA;assign NA  to -1 and -2
def notNA(x):
    if x == -1 or x == -2:
        return False
    return True
def isNA(x):
    if x == -1 or x == -2:
        return True
    return False
train_x.where(train_x.applymap(notNA), np.nan, inplace = True)
test_x.where(test_x.applymap(notNA), np.nan, inplace = True)
# missrate
# line missrate:
#train_x_NA = train_x[np.isnan(train_x), axis = 1)] # 15000 rows
#test_x_NA = test_x[np.isnan(test_x), axis = 1)] # 5000 rows
# feature missrate:
#trainMissrate = getMissrate(train_x) #max:0.9678
#testMissrate = getMissrate(test_x) #max: 0.9648

trFeatureMiss = train_x.apply(getMissrate, reduce = False).to_frame()
trFeatureMiss.columns = ['trMissrate']
trFeatureMiss = pd.merge(featuresType, trFeatureMiss, left_on = 'feature',
                         right_index= True)
teFeatureMiss = test_x.apply(getMissrate, reduce = False).to_frame()
teFeatureMiss.columns = ['teMissrate']
teFeatureMiss = pd.merge(featuresType, teFeatureMiss,  left_on='feature',
                         right_index= True)
FeatureMiss = pd.merge(trFeatureMiss, teFeatureMiss)

## has 642 feture has same missrate 0.0012666  ###???
trFeatureMissInfo = trFeatureMiss.groupby(by = ['trMissrate']).count()

# fig_num = 0
# def missrate_feanum(df, miss_col,file_name):
#     rate = 0.0
#     nums = [0] * 100
#     num = 0
#     temp = df
#     for i in range(100):
#         rate = i * 0.01
#         num = np.sum(temp[miss_col] <= rate)
#         if(i > 0):
#             nums[i] = nums[i-1] + num
#         else:
#             nums[i] = num
#         temp = temp[temp[miss_col] > rate]
#     global fig_num
#     plt.figure(fig_num)
#     fig_num += 1
#     plt.plot(nums,'-*')
#     #plt.show()
#     plt.savefig(fig_path + file_name + '.png')
# trNumMiss = trFeatureMiss[trFeatureMiss['type'] == 'numeric']
# missrate_feanum(trNumMiss,'trMissrate','tr_miss_num')
# trCatMiss = trFeatureMiss[trFeatureMiss['type'] == 'category']
# missrate_feanum(trCatMiss,'trMissrate','tr_miss_cat')
# teNumMiss = teFeatureMiss[teFeatureMiss['type'] == 'numeric']
# missrate_feanum(teNumMiss,'teMissrate','te_miss_num')
# teCatMiss = teFeatureMiss[teFeatureMiss['type'] == 'category']
# missrate_feanum(teCatMiss,'teMissrate','te_miss_cat')

FeatureMiss = FeatureMiss[['feature', 'type', 'trMissrate', 'teMissrate']]
for row in range(len(FeatureMiss)):
    #if FeatureMiss.ix[row, 'type'] == 'category':
    if FeatureMiss.ix[row,'trMissrate'] >= rem_threshold:
        feature = FeatureMiss.ix[row, 'feature']
        train_x.drop(feature, axis = 1, inplace = True)
        test_x.drop(feature, axis = 1, inplace = True)

# As xgboost is treat every variable as numeric and support for miss
#feat0: fill NA to - 1 for category
if feat_name == 'feat0':
    for feature in train_x.columns[1:]:
        if np.any(featuresType.loc[featuresType['feature'] == feature, 'type'] == 'category'):
            train_x[feature].fillna(-1, inplace = True)
            test_x[feature].fillna(-1, inplace = True)
train = pd.merge(train_x, train_y)
#As regression, create dummy variable for category
sigVal = -100
test_x['y'] = sigVal
datas = pd.concat([train,test_x])
datas = catToDummy(data=datas, featuresType =featuresType)
train = datas[datas['y'] != sigVal]
test = datas[datas['y'] == sigVal]
test.drop(labels=['y'], axis = 1, inplace = True)

# divide train to train_train for train and train_valid for model ensemble
train_train_percent = 0.9
rng = np.random.RandomState(2016)
num_train = len(train)
indexs = rng.permutation(num_train)
train_train_num = (int)(num_train*train_train_percent)
train_train = train.iloc[indexs[0:train_train_num],:]
train_valid = train.iloc[indexs[train_train_num:],:]

#output
file_ids = {'train':train_train, 
            'train_valid':train_valid,
            'train_all':train,
            'test':test}
for file_id in file_ids:
    file_name = output_path + file_id + '_' + feat_name + '.csv'
    df = file_ids[file_id]
    df.to_csv(file_name,indexs=False)
    # 'nonan' denote:对于数值属性,用平均值填充
    file_name = output_path + file_id + '_' + feat_name + '_nonan' + '.csv'
    df.fillna(df.mean(),inplace=True)
    df.to_csv(file_name,indexs=False)


#train = pd.read_csv(o_path + 'train_' + F_name + ('_nonan' if remove_nan else '') + '.csv')
#test = pd.read_csv(o_path + 'test_' + F_name + ('_nonan' if remove_nan else '') + '.csv')

#gen_feature(F_name,i_path, o_path)

# cat_file_name = output_path + feat_name + '_catName' + ('_nonan' if remove_nan else '') + '.csv'
# cat_feature_names = datas.columns[1047:]
# handler = open(cat_file_name,'w')
# writer = csv.writer(handler)
# writer.writerows(cat_feature_names)
# handler.flush()
options(error = dump.frames)
library(xgboost)
# library(Matrix)
# read data
setwd('/Users/HAN/Documents/CashBus')
iDir <- './input/'
train <- read.csv(paste0(iDir,'train_x.csv'))
test <- read.csv(paste0(iDir,'test_x.csv'))
train.y <- read.csv(paste0(iDir,'train_y.csv'))
ft <- read.csv(paste0(iDir,'features_type.csv'))
fn.cat=as.character(ft[ft[,2]=='category',1])

fn.num=as.character(ft[ft[,2]=='numeric',1])


# create dummy variables
temp.train=data.frame(rep(0,nrow(train)))
temp.test=data.frame(rep(0,nrow(test)))
for(f in fn.cat){
  levels=unique(train[,f])
  col.train=data.frame(factor(train[,f],levels=levels))
  col.test=data.frame(factor(test[,f],levels=levels))
  colnames(col.train)=f
  colnames(col.test)=f
  temp.train=cbind(temp.train,
                   model.matrix(as.formula(paste0('~',f,'-1')),data=col.train))
  temp.train[,paste0(f,'-1')]=NULL
  temp.test=cbind(temp.test,
                  model.matrix(as.formula(paste0('~',f,'-1')),data=col.test))
  temp.test[,paste0(f,'-1')]=NULL
}
temp.train[,1]=NULL
temp.test[,1]=NULL
train.new=Matrix::Matrix(data.matrix(cbind(train[,c('uid',fn.num)],temp.train)),sparse=T)
test.new=Matrix::Matrix(data.matrix(cbind(test[,c('uid',fn.num)],temp.test)),sparse=T)

# fit xgboost model

dtrain=xgb.DMatrix(data=train.new[,-1],label=1-train.y$y)
dtest= xgb.DMatrix(data=test.new[,-1])

model=xgb.train(booster='gbtree',
                objective='binary:logistic',
                scale_pos_weight=8.7,
                gamma=0,
                lambda=700,
                subsample=0.7,
                colsample_bytree=0.30,
                min_child_weight=5,
                max_depth=8,
                eta=0.02,
                data=dtrain,
                nrounds=1520,
                metrics='auc',
                nthread=2)

# predict probabilities
pred=1-predict(model,dtest)

# 现在可以提交结果了，下面的问题就是调参数，欢迎大家来rnorm.com讨论机器学习问题！
write.csv(data.frame('uid'=test.new.1[,1],'score'=pred),file='rnorm.com.csv',row.names=F)
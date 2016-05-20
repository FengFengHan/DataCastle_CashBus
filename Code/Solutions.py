import sys
sys.path.extend(['/Users/HAN/AllCode/Projects/CashBus'])
from Code.Models import *
from Code.MyUtil import *

class MySolution:
    def __init__(self,model,feat_name):
        self.model_ = model
        self.feat_name_ = feat_name
        self.id_ = model.id_ + '_' + feat_name

    def param_sel_var(self,param, var,start,end,step,id_):
        model = self.model_
        feat_name = self.feat_name_
        train = pd.read_csv(feat_path + 'train_'+  feat_name + '.csv')
        train_x = train.drop(labels=['uid','y'],axis=1)
        train_y = train.y
        png_name = fig_path
        png_name += ('%s_%s_%d.png' % (self.id_,var,id_))
        log_csv_name = ('%sT%s_%s.csv' % (log_write_path,
                                                      time.strftime("%m%d_%H%M", time.localtime()),
                                          self.id_
                                                      ))
        model.param_sel_var(var, start, end, step, param, train_x, train_y, png_name, log_csv_name)

    def param_sel_space(self,param_space):
        model = self.model_
        feat_name = self.feat_name_
        train = pd.read_csv(feat_path + 'train_' + feat_name + '.csv')
        train_x = train.drop(labels=['uid', 'y'], axis=1)
        train_y = train.y
        log_csv_name = ('%sT%s_%s.csv' % (log_write_path,
                                               time.strftime("%m%d_%H%M", time.localtime()),
                                               self.id_))
        model.param_sel_space(param_space, train_x, train_y, log_csv_name)

#feat_name = 'feat2'
#model = SklRige()
#parameter_sel_grid(model,'alpha',2.5,3.3,0.1,3)
#model = SklLrl2()
#parameter_sel_grid(model,'C', 2e-4, 3e-4, 1e-5, 5)
#model = SklLrl1()
#parameter_sel_grid(model, 'C', 2e-2-1e-3, 2e-2 + 1e-3, 1e-4,3)
#model = SklLasso()
#parameter_sel_grid(model,'alpha',38e-6,40e-6,2e-7,3)

# feat_name = 'feat2'
# # model = XgbTree()
# # parameter_sel_hyper(model,feat_name = 'feat3')
# model = SklLrl1()
# model.param_ = {'C': 0.0208}
# train = pd.read_csv(feat_path + 'train_all_' + feat_name + feat_na.get(model.id_,'_nonan') + '.csv' )
# train_x = train.drop(labels=['uid','y'], axis=1)
# train_y = train.y
# model.cv_run(train_x,train_y,1,9)

#model = XgbTree()
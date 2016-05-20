import sys
sys.path.extend(['/Users/HAN/AllCode/Projects/CashBus'])
import pickle
from Code.Solutions import *

class MyAnswer():
    def __init__(self,solution,param, score,myid =0):
        self.model_ = solution.model_
        self.feat_name_ = solution.feat_name_
        self.param_ = param
        self.score_ = score
        self.id_ = solution.id_ + "_"+ str(myid)
        self.weight_ = -1

#select solutions and sort by score
solution_sels = [MySolution(XgbTree(),'feat3'),
             MySolution(SklLrl1(),'feat2_nonan'),
             MySolution(SklLasso(), 'feat2_nonan')]

#from solutions to answers
def get_answers(solution_sels):
    answers = []
    for solution_sel in solution_sels:
        solution_logs = pd.read_csv(sol_sel_path + '%s.csv' %solution_sel.id_)
        for i in range(solution_logs.shape[0]):
            score = (solution_logs.iloc[i,:]).score_mean
            param = (solution_logs.iloc[i,:]).to_dict()
            answer = (solution_sel, param,score,i)
            answers.append(answer)
    answers.sort(key=lambda x: x.score_, reverse=True)
    return answers

answers = get_answers(solution_sels)
for answer in answers:
    print('train_cv_score: %.6f' %answer.score_)

# predict train_valid by answers
valid_bag_rate = 0.9
valid_num = 10
def predict_train_valid(answers_):
    scores = np.zeros(valid_num)
    for answer_count in range(len(answers_)):
        answer = answers_[answer_count]
        valid = pd.read_csv(feat_path + 'train_valid_%s.csv'
                            % (answer.feat_name_))
        train = pd.read_csv(feat_path + 'train_%s%s.csv'
                            % (answer.feat_name_) )
        train_x = train.drop(labels=['uid', 'y'], axis=1)
        train_y = train.y
        valid_indexss = get_divides(len(valid), valid_bag_rate, valid_num)

        print("anwer_id_: %s" %answer.id_)
        for valid_count in range(valid_num):
            valid_indexs = valid_indexss[valid_count]
            valid_cur = valid.iloc[valid_indexs, :]
            valid_x = valid_cur.drop(labels=['uid','y'],axis= 1)
            pred = answer.model_.run(train_x,train_y,valid_x)
            pred = scale(pred) ## to ensemble result of different model
            valid_y = valid_cur.y
            scores[valid_count] = metrics.roc_auc_score(y_true=valid_y, y_score=pred)
            df = pd.DataFrame(columns=['ori','pred'])
            df.ori = valid_y
            df.pred = pred
            df.to_csv(ensemble_path + 'solution_%s_%d.csv' % (answer.id_, valid_count), index=False)
        print(' before ensemble: score_ave %.6f, score_std %.6f' %(np.mean(scores), np.std(scores)))


def get_predict_results(answers_):
    #get y_lens for every valid
    y_len_max = 0
    y_lens = np.zeros(valid_num, 'int')
    for valid_count in range(valid_num):
        y = pd.read_csv(ensemble_path + 'solution_%s_%d.csv' % (answers_[0].id_, valid_count))
        y_length = len(y)
        y_lens[valid_count] = y_length
        if (y_length > y_len_max):
            y_len_max = y_length

    # get y_true for every valid
    y_trues = np.zeros((y_len_max, valid_num), 'int')
    for valid_count in range(valid_num):
        y = pd.read_csv(ensemble_path + 'solution_%s_%d.csv' % (answers_[0].id_, valid_count))
        y_trues[:y_lens[valid_count], valid_count] = y.ori

    # get y_pred for every valid and solution
    answer_num = len(answers_)
    y_predss = np.zeros((y_len_max, valid_num, answer_num))
    for answer_count in range(answer_num):
        for valid_count in range(valid_num):
            y = pd.read_csv(ensemble_path + 'solution_%s_%d.csv' % (answers_[answer_count].id_, valid_count))
            y_predss[:y_lens[valid_count], valid_count, answer_count] = y.pred

    return (y_lens,y_trues,y_predss)

def get_ens_weight_greedy(anwers_):
    y_lens, y_trues, y_predss = get_predict_results(anwers_)
    #init
    answer_num = len(anwers_)
    ws = np.zeros(answer_num)
    score_ave_max = 0
    pred_ens = y_predss[:,:,0]
    w_ens = 1
    ws[0] = 1
    scores = np.zeros(valid_num)
    for valid_count in range(valid_num):
        pred_ = pred_ens[:,valid_count]
        scores[valid_count] = metrics.roc_auc_score(
            y_true=y_trues[:y_lens[valid_count],valid_count],
            y_score=pred_[:y_lens[valid_count]]
        )
    score_ave_max = np.mean(scores)
    # add one by one
    step = 0.001
    for answer_count in range(1,answer_num):
        best_w = 0
        w = 0
        while w <= 1:
            for valid_count in range(valid_num):
                pred_cur = y_predss[:,valid_count,answer_count]
                pred_ = (pred_ens[:, valid_count] + w * pred_cur) / (w_ens + w)
                scores[valid_count] = metrics.roc_auc_score(
                                            y_true=y_trues[:y_lens[valid_count],valid_count],
                                            y_score=pred_[:y_lens[valid_count]]
                                              )
            score_ave = np.mean(scores)
            if( score_ave > score_ave_max):
                score_ave_max = score_ave
                best_w  = w
            w += step
        print("best_w: %.2f, score_ave_max: %.6f" %(best_w,score_ave_max))
        #update w_ens and pred_ens
        w_ens += best_w
        pred_ens = pred_ens + best_w * y_predss[:,:,answer_count]
        ws[answer_count] = best_w

    return (ws,score_ave_max)
# ws = [ 1.     0.999  0.926  0.999  0.966  0.999  0.998  0.31   0.926]
weights,score_ave = get_ens_weight_greedy(answers)
for i in range(len(answers)):
    answers[i].weight_ = weights[i]

f = open(answer_path + '%s_answer_%d.txt' %(time.strftime("%m%d_%H%M",time.localtime()),
                                            int(score_ave*10000) ), 'wb' )
pickle.dump(answers,f)


def ens_result(answers_,ws):
    y_lens, y_trues, y_predss = get_predict_results(answers_)
    answer_num = len(answers)
    y_len_max = (y_predss[:,0,0]).shape[0]
    ens_preds = np.zeros((y_len_max,valid_num))
    scores = np.zeros(valid_num)
    for valid_count in range(valid_num):
        for answer_count in range(answer_num):
            ens_preds[:,valid_count] += y_predss[:,valid_count,answer_count]*ws[answer_count]
        pred_ = ens_preds[:,valid_count]/np.sum(ws)
        scores[valid_count] = metrics.roc_auc_score(y_true=y_trues[:y_lens[valid_count],valid_count],
                                                    y_score=pred_[:y_lens[valid_count]])
    score_ave = np.mean(scores)
    return score_ave

ws1 = np.ones(len(answers))
score1 = ens_result(answers,ws1)
print('score ave ensemble: %.6f' %score1)
# ws2 = np.array([1.0, 0.0, 0.276, 0.179])
# score2 = ens_result(len(solutions),valid_num,y_lens,y_predss,y_trues,ws2)



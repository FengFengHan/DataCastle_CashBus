
# cv test for model
# def model_cv_res(param, model_name):
#     time_start = datetime.datetime.now()
#     num_train = train.shape[0]
#     num_valid = num_train//n_folds
#     score = np.zeros((n_runs, n_folds),dtype=float)
#     print('------------------------------------')
#     print(param)
#     for run in range(n_runs):
#         print('run: %d' %run)
#         rng = np.random.RandomState(2016 + 100 *run)
#         indexs = rng.permutation(num_train)
#         for fold in range(n_folds):
#             time_start_fold = datetime.datetime.now()
#             valid_index_start = fold*num_valid
#             valid_index_end = (fold +1)*num_valid
#             valid_indexs = indexs[valid_index_start:valid_index_end]
#             cv_valid = train.iloc[valid_indexs,:]
#             train_indexs = np.concatenate((indexs[:valid_index_start],
#                                           indexs[valid_index_end:]))
#             cv_train = train.iloc[train_indexs,:]
#
#             #evals_result = {}
#             d_cv_train = xgb.DMatrix(cv_train.drop(labels=['uid','y'],axis = 1),
#                                      label =cv_train.y, missing=np.nan)
#             d_cv_valid = xgb.DMatrix(cv_valid.drop(labels=['uid','y'],axis = 1),
#                                      label =cv_valid.y, missing=np.nan)
#             watchlist = [(d_cv_valid, 'valid')]
#             #watchlist = [(d_cv_valid, 'valid')]
#             bst = xgb.train(param,d_cv_train,num_boost_round=param['num_round'],
#                             evals = watchlist, verbose_eval=False)
#             # if debug:
#             #     score_train = evals_result['train']['auc']
#             #     score_val = evals_result['valid']['auc']
#             #     fig = plt.figure(fold)
#             #     ax1 = fig.add_subplot(211)
#             #     ax1.plot(score_train[50:],'b')
#             #     ax2 = fig.add_subplot(212)
#             #     ax2.plot(score_val[50:],'r')
#             #     save_name = test_path + ('eta_round_%.1f_%run_%d' %(param['eta'],run, fold))
#             #     fig.savefig(save_name + '.png')
#             #     data_handler = open(save_name + '.csv','w')
#             #     data_writer = csv.writer(data_handler)
#             #     data_writer.writerow(score_train)
#             #     data_writer.writerow(score_val)
#             #     data_handler.flush()
#             pred = bst.predict(d_cv_valid, ntree_limit=bst.best_ntree_limit)
#             score[run][fold] = metrics.roc_auc_score(y_true=cv_valid.y, y_score=pred)
#             fold_time = str((datetime.datetime.now() - time_start_fold).seconds)
#             print('fold: %d, score: %.6f, time: %ss' %(fold, score[run][fold], fold_time))
#             fold += 1
#             cur_save_path = save_cv_path + ('run%d_fold%d/' %(run,fold))
#             if not os.path.exists(cur_save_path):
#                 os.makedirs(cur_save_path)
#             file_name = cur_save_path + model_name + ".csv"
#             cv_res = pd.DataFrame(columns=['y_pred','y_label'])
#             cv_res.y_pred = pred
#             cv_res.y_label = cv_valid.y
#             cv_res.to_csv(file_name,index=False)
#         print('run: %d, score: %.6f' %(run, np.mean(score[run])))
#     score_mean = np.mean(score)
#     score_std = np.std(score)
#     print('mean score: %.6f, std: %.6f' %(score_mean, score_std))


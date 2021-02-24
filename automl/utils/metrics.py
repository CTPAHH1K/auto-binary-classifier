from sklearn.metrics import roc_auc_score

metrics_list = ['ROCAUC']

score_func = {'ROCAUC': roc_auc_score}

higher_better = {'ROCAUC': True}
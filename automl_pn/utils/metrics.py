from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, precision_score,
                             recall_score)

# TODO add more metrics
metrics_list = ['roc_auc', 'accuracy', 'f1', 'precision', 'recall']

score_func = {'roc_auc': roc_auc_score,
              'accuracy': accuracy_score,
              'f1': f1_score,
              'precision': precision_score,
              'recall': recall_score,
              }

higher_better = {'roc_auc': True,
                 'accuracy': True,
                 'f1': True,
                 'precision': True,
                 'recall': True,
                 }

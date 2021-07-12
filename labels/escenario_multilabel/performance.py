import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import pandas as pd
import numpy as np
from config import corpus, model_label as model, percent, n_samples, multilabel_dict
from collections import Counter
from BERTclassifier import getTopics

mode = multilabel_dict.split('_')[0]
cluster_set = list(dict.fromkeys(getTopics(model)))

oracle = pd.read_csv('labels/escenario_multilabel/multilabel_dicts/%s_multilabel_%s_100000_%s.csv'%(mode, model, percent), header=None, names=['cluster', 'label'])
pred = pd.read_csv('labels/escenario_multilabel/multilabel_dicts/%s.csv'%multilabel_dict, header=None, names=['cluster', 'label'])

df = pd.DataFrame(index=cluster_set, columns=['oracle', 'pred'])

for cluster in cluster_set:
    df.loc[cluster] = [oracle[oracle['cluster'] == cluster]['label'].item(), pred[pred['cluster'] == cluster]['label'].item()]

pred_ml = 0
real_ml = 0
bad_pred_multilabel = []
good_pred_multilabel = []
for cluster in cluster_set:
   if df.loc[cluster, 'pred'].split(' ').__len__() == 2:
       pred_ml += 1
   if df.loc[cluster, 'oracle'].split(' ').__len__() == 2:
       real_ml += 1
   if df.loc[cluster, 'pred'].split(' ').__len__() == 2 and df.loc[cluster, 'oracle'].split(' ').__len__() == 1:
       bad_pred_multilabel.append(cluster)
   if df.loc[cluster, 'pred'].split(' ').__len__() == 2 and df.loc[cluster, 'oracle'].split(' ').__len__() == 2:
       good_pred_multilabel.append(cluster)


ml_score = (pred_ml/real_ml)
correct_ml_preds = (len(good_pred_multilabel)/real_ml)

print(pred_ml, len(bad_pred_multilabel))
print(ml_score, correct_ml_preds)
topics = getTopics(model)
topics = Counter(topics)
muestras_afectadas = 0
for cluster in bad_pred_multilabel:
     muestras_afectadas += topics.get(cluster)
print(muestras_afectadas)

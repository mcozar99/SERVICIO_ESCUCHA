import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import numpy as np
import pandas as pd
from config import model_label as model, corpus, n_samples, probabilistic_mode, random_mode, percent
from BERTclassifier import getTrueLabels, getTopics, getProbabilities
from collections import Counter
import csv
import random

cluster_set = list(dict.fromkeys(getTopics(model)))
cluster_dict = pd.DataFrame(index =cluster_set, columns=['pred_monolabel', 'pred_multilabel'])

df = pd.DataFrame(getTopics(model), columns = ['cluster'])
df['true'] = getTrueLabels(corpus)

if probabilistic_mode:
   probs= getProbabilities(model, 'df')
   for cluster in cluster_set:
       if cluster == -1:
           cluster_dict.loc[cluster] = ['descarte', 'descarte']
           continue
       data = df[df['cluster'] == cluster]
       aux = list(probs.iloc[data.index][cluster].nlargest(n_samples).index)
       data = Counter(data.loc[aux].true.tolist())
       monolabel = list(data.keys())[0]
       if data.__len__() > 1:
           if list(data.values())[1] / n_samples >= percent/100:
               multilabel = '%s %s'%(list(data.keys())[0], list(data.keys())[1])
           else:
               multilabel = list(data.keys())[0]
       else:
           multilabel = list(data.keys())[0]
       cluster_dict.loc[cluster] = [monolabel, multilabel]
   cluster_dict["pred_monolabel"].to_csv('./labels/escenario_monolabel/label_dict/probabilities_dict_%s.txt'%model, index=True, header=None)
   cluster_dict["pred_multilabel"].to_csv('./labels/escenario_multilabel/multilabel_dicts/probabilities_multilabel_%s_%s.csv'%(model,percent),index=True, header=None)
   cluster_dict = pd.DataFrame(index =cluster_set, columns=['pred_monolabel', 'pred_multilabel'])

if random_mode:
    for cluster in cluster_set:
        if cluster == -1:
            cluster_dict.loc[cluster] = ['descarte', 'descarte']
            continue
        data = Counter(df[df['cluster'] == cluster].sample(n_samples)['true'].tolist())
        monolabel = list(data.keys())[0]
        if data.__len__() > 1:
            if list(data.values())[1] / n_samples >= percent/100:
                multilabel = '%s %s'%(list(data.keys())[0], list(data.keys())[1])
            else:
                multilabel = list(data.keys())[0]
        else:
            multilabel = list(data.keys())[0]
        cluster_dict.loc[cluster] = [monolabel, multilabel]
    cluster_dict["pred_monolabel"].to_csv('./labels/escenario_monolabel/label_dict/random_dict_%s.txt'%model, index=True, header=None)
    cluster_dict["pred_multilabel"].to_csv('./labels/escenario_multilabel/multilabel_dicts/random_multilabel_%s_%s.csv'%(model,percent),index=True, header=None)
    cluster_dict = pd.DataFrame(index =cluster_set, columns=['pred_monolabel', 'pred_multilabel'])


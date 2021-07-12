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
from subprocess import call
import os.path

if not os.path.exists('./results/%s/evaluation'%model):
    call('mkdir ./results/%s/evaluation'%model, shell=True)
if not os.path.exists('./results/%s/labels'%model):
    call('mkdir ./results/%s/labels'%model, shell=True)
if not os.path.exists('./results/%s/labels/multilabel'%model):
    call('mkdir ./results/%s/labels/multilabel'%model, shell=True)
if not os.path.exists('./results/%s/labels/monolabel'%model):
    call('mkdir ./results/%s/labels/monolabel'%model, shell=True)

cluster_set = list(dict.fromkeys(getTopics(model)))
cluster_dict = pd.DataFrame(index =cluster_set, columns=['pred_monolabel', 'pred_multilabel'])

aux = df = pd.read_csv('./corpus/preprocessed/preprocess_%s'%corpus, delimiter='\t', header=None, names=['index', 'true', 'text'], quoting=csv.QUOTE_NONE, error_bad_lines=False)
df = pd.DataFrame(getTopics(model), columns = ['cluster'])
df['index'] = aux['index']
df['true'] = getTrueLabels(corpus)
df['text'] = aux['text']



i = 0

if probabilistic_mode:
   acumulador = 0
   probs= getProbabilities(model, 'df')
   for cluster in cluster_set:
       if cluster == -1:
           cluster_dict.loc[cluster] = ['descarte', 'descarte']
           continue
       data = df[df['cluster'] == cluster]

       cluster_probs = probs.loc[data.index, cluster]

       cluster_probs = cluster_probs.nsmallest(n_samples)
       """ MODELO PARA LAS MEJORES PROBABILIDADES
       if cluster_probs[cluster_probs == 1].__len__() > n_samples:
           cluster_probs = cluster_probs[cluster_probs == 1].sample(n_samples, random_state=7)
       else:
           cluster_probs = cluster_probs.nlargest(n_samples)
       """
       aux = data.loc[cluster_probs.index, 'true']
       data = Counter(aux)
       acumulador += sum(list(data.values()))
       monolabel = max(data, key=data.get)

       if data.__len__() > 1:
           if sorted(list(data.values()))[-2] / sum(data.values()) >= percent/100:
               maxima = max(data, key=data.get)
               del[data[maxima]]
               s_maxima =  max(data, key=data.get)
               multilabel = '%s %s'%(maxima, s_maxima)
           else:
               multilabel = max(data, key=data.get)
       else:
           multilabel = max(data, key=data.get)
       cluster_dict.loc[cluster] = [monolabel, multilabel]

       i += 1
       #if i == 10:
       #    break

   cluster_dict["pred_monolabel"].to_csv('./labels/escenario_monolabel/label_dict/probabilities_dict_%s_%s_%s.txt'%(model, n_samples, percent), index=True, header=None)
   cluster_dict["pred_multilabel"].to_csv('./labels/escenario_multilabel/multilabel_dicts/probabilities_multilabel_%s_%s_%s.csv'%(model, n_samples, percent),index=True, header=None)
   cluster_dict = pd.DataFrame(index =cluster_set, columns=['pred_monolabel', 'pred_multilabel'])

if random_mode:
    acumulador = 0
    analisis_multilabel=[]
    for cluster in cluster_set:
        if cluster == -1:
            cluster_dict.loc[cluster] = ['descarte', 'descarte']
            continue
        if n_samples > df[df['cluster'] == cluster].shape[0]:
            data = Counter(df[df['cluster'] == cluster]['true'].tolist())
        else:
            data = Counter(df[df['cluster'] == cluster].sample(n_samples, random_state=7)['true'].tolist())

        acumulador += sum(list(data.values()))
        monolabel = max(data, key=data.get)
        if data.__len__() > 1:
            if sorted(list(data.values()))[-2] / sum(data.values()) >= percent/100:
                maxima = max(data, key=data.get)
                del[data[maxima]]
                s_maxima =  max(data, key=data.get)
                multilabel = '%s %s'%(maxima, s_maxima)
                analisis_multilabel.append(multilabel)
            else:
                multilabel = max(data, key=data.get)
        else:
            multilabel = max(data, key=data.get)
        cluster_dict.loc[cluster] = [monolabel, multilabel]
    cluster_dict["pred_monolabel"].to_csv('./labels/escenario_monolabel/label_dict/random_dict_%s_%s_%s.txt'%(model, n_samples, percent), index=True, header=None)
    cluster_dict["pred_multilabel"].to_csv('./labels/escenario_multilabel/multilabel_dicts/random_multilabel_%s_%s_%s.csv'%(model,n_samples, percent),index=True, header=None)
    cluster_dict = pd.DataFrame(index =cluster_set, columns=['pred_monolabel', 'pred_multilabel'])

print('WE TOOK %s SAMPLES TO MAKE THIS PROBS AND RANDOM DICTIONARIES'%acumulador, flush=True)
print(Counter(analisis_multilabel))
print(len(analisis_multilabel))


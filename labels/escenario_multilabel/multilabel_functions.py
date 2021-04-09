import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix
from config import model_label as model, corpus, percent
from BERTclassifier import getTrueLabels, getTopics
from collections import Counter
import os.path
from subprocess import call

def importance_cluster_list(clusters, cluster, true_labels):
    labels_of_cluster = []
    for i in range(len(clusters)):
        if cluster == clusters[i]:
           labels_of_cluster.append(true_labels[i])
    c = Counter(labels_of_cluster)
    s = sum(c.values())
    return [(i, c[i] / s * 100.0) for i, count in c.most_common()]

def multilabel_dictionary(corpus, model, second_label_importance):
    true_labels = getTrueLabels(corpus)
    clusters = getTopics(model)
    n_clusters = list(dict.fromkeys(clusters))
    if not os.path.exists('./labels/escenario_multilabel/multilabel_dicts'):
        call('mkdir ./labels/escenario_multilabel/multilabel_dicts', shell=True)
    f = open('./labels/escenario_multilabel/multilabel_dicts/multilabel_%s_%s_dict.csv'%(model, second_label_importance), 'w', encoding='utf-8')
    for cluster in n_clusters:
        importance = importance_cluster_list(clusters, cluster, true_labels)
        #print('CLUSTER %s'%cluster)
        #print(importance)
        if cluster == -1:
            f.write('%s,descarte\n'%cluster)
            continue
        if len(importance) == 1:
            f.write('%s,%s\n'%(cluster,importance[0][0]))
            continue
        if importance[1][1] > second_label_importance:
            f.write('%s,%s %s\n'%(cluster,importance[0][0], importance[1][0]))
        else:
            f.write('%s,%s\n'%(cluster,importance[0][0]))
    f.close()


def get_multilabel_dict(model, second_label_importance):
    return pd.read_csv('./labels/escenario_multilabel/multilabel_dicts/multilabel_%s_%s_dict.csv'%(model, second_label_importance), delimiter=',', names=['cluster', 'label'])


def label_samples(model, second_label_importance):
    clusters = getTopics(model)
    multilabel_dict = get_multilabel_dict(model, second_label_importance)
    multilabel_dict = dict(zip(multilabel_dict.cluster, multilabel_dict.label))
    f = open('./results/%s/multilabel_predictions_%s.txt'%(model, second_label_importance), 'w', encoding='utf-8')
    for cluster in clusters:
        f.write(multilabel_dict.get(cluster) + '\n')
    f.close()


def get_multilabel_predictions(model, second_label_importance):
    return list(pd.read_csv('./results/%s/multilabel_predictions_%s.txt'%(model, second_label_importance), header=None)[0].to_numpy())


def get_accuracy(model, second_label_importance):
    true = getTrueLabels(corpus)
    pred = get_multilabel_predictions(model, second_label_importance)
    correct_indexes = []
    for i in range(len(true)):
        if true[i] in pred[i]:
            correct_indexes.append(i)
    acc = len(correct_indexes)/len(true)
    pure_acc = len(correct_indexes)/(len(true) - Counter(pred).get('descarte'))
    print('ACCURACY IS: %s'%acc)
    print('ACCURACY WITHOUT DISCARDS: %s'%pure_acc)

def accuracy_per_label(model, second_label_importance):
    true = getTrueLabels(corpus)
    pred = get_multilabel_predictions(model, second_label_importance)
    label_set = list(dict.fromkeys(true))
    df = pd.DataFrame(index=label_set, columns = ['Discards of topic', '% discard', 'Correct', 'Total', 'Accuracy', 'Accuracy Without Discards'])
    for label in label_set:
         total = Counter(true).get(label)
         correct = 0
         discards_of_topic = 0
         for i in range(len(true)):
             if true[i] in pred[i] and label in true[i]:
                 correct+=1
             if true[i] in label and 'descarte' in pred[i]:
                 discards_of_topic +=1
         perc_discard = (discards_of_topic/total)
         acc = (correct/total)
         pure_acc = (correct/(total-discards_of_topic))
         df.loc[label] = [discards_of_topic, '{:.2%}'.format(perc_discard), correct, total, '{:.2%}'.format(acc), '{:.2%}'.format(pure_acc)]
    print(df.to_string())


multilabel_dictionary(corpus, model, percent)
label_samples(model, percent)
accuracy_per_label(model, percent)
get_accuracy(model, percent)

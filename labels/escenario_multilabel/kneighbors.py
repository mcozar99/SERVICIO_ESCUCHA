import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from config import corpus, model_label as model, percent, sentence_transformer, kneighbors_eval, kneighbors_labeling
from sentence_transformers import SentenceTransformer
from BERTclassifier import getEmbeddings, getTrueLabels, getTopics, loadPreprocessedText
from datetime import datetime
from xlrd import open_workbook
from xlutils.copy import copy
import xlwt

df = pd.read_csv('results/%s/labels/multilabel/multilabel_predictions_%s.txt'%(model, percent), names=['pred'], header=None)
df['true'] = getTrueLabels(corpus)
df['text'] = loadPreprocessedText(corpus)
df = df.assign(code=[*getEmbeddings(model, 'numpy')])

label_set = list(dict.fromkeys(getTrueLabels(corpus)))

def kneighbors(df):
    discards = df[df['pred'] == 'descarte']
    df.drop(df[df['pred'] == 'descarte'].index, inplace=True)
    print(df)
    for item in discards.iterrows():
        distance = cosine_similarity([item[1]['code']], df['code'].tolist())
        df['distance'] = distance[0]
        item[1]['pred'] = df[df.distance == df.distance.max()]['pred'].tolist()[0]
    frames = [df, discards]
    df = pd.concat(frames).sort_index(axis=0)
    print(df)
    df['pred'].to_csv('./results/%s/labels/multilabel/kneighbors_multilabel_predicts.txt'%model, index=False, header=False)

def get_kneighbors_predictions():
    preds = pd.read_csv('./results/%s/labels/multilabel/kneighbors_multilabel_predicts.txt'%model, names=['pred'],header=None)
    preds['true'] = getTrueLabels(corpus)
    return preds

def get_kneighbors_discard_predictions():
    preds = pd.read_csv('./results/%s/labels/multilabel/kneighbors_multilabel_predicts.txt'%model, names=['pred'],header=None)
    preds['true'] = getTrueLabels(corpus)
    preds['cluster'] = getTopics(model)
    preds = preds[preds['cluster'] == -1]
    return preds


def evaluation(true, pred):
    tot = Counter(true)
    for label in label_set:
        if label not in tot.keys():
            tot.update({label : 0})
    evaluation = pd.DataFrame(index=label_set, columns=['precision', 'recall', 'f1'])
    evaluation['Total']=tot.values()
    for label in label_set:
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        for i in range(len(true)):
            if label == true[i] and true[i] in pred[i]:
                true_positives += 1
            if label == true[i] and true[i] not in pred[i]:
                false_negatives += 1
            if label in pred[i] and true[i] not in pred[i]:
                false_positives += 1
            if label != true[i] and true[i] not in pred[i]:
                true_negatives += 1
        topic_precision = 0
        topic_recall = 0
        topic_f1 = 0
        if true_positives + false_positives != 0:
            topic_precision = true_positives / (true_positives + false_positives)
        if true_positives + false_negatives != 0:
            topic_recall = true_positives / (true_positives + false_negatives)
        if topic_precision + topic_recall != 0:
            topic_f1 = 2 * (topic_precision * topic_recall) / (topic_precision + topic_recall)
        evaluation.loc[label] = [topic_precision, topic_recall, topic_f1, tot.get(label)]
    total = len(true)
    final_numbers = []
    prec, recall, f1 = (0,0,0)
    for eval in evaluation.iterrows():
        prec += (eval[1]['precision'] * eval[1]['Total'])/total
        recall += (eval[1]['recall'] * eval[1]['Total'])/total
        f1 += (eval[1]['f1'] * eval[1]['Total'])/total
    i = 0
    for j in range(len(true)):
        if true[j] in pred[j]:
            i += 1
    acc = i / sum(tot.values())
    print('Acc: %s \t Prec: %s \t Recall: %s \t F1: %s'%(acc,prec,recall,f1))


if kneighbors_labeling:
    kneighbors(df)
if kneighbors_eval:
    print('KNEIGHBORS EVALUATION')
    print('DISCARDS EVALUATION')
    discards_pred = get_kneighbors_discard_predictions()
    evaluation(discards_pred['true'].tolist(), discards_pred['pred'].tolist())
    print('FINAL EVALUATION')
    final_pred = get_kneighbors_predictions()
    evaluation(final_pred['true'].tolist(), final_pred['pred'].tolist())

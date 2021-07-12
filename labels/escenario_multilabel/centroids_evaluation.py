import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from config import corpus, model_label as model, percent, sentence_transformer, n_samples, multilabel_dict
from sentence_transformers import SentenceTransformer
from BERTclassifier import getEmbeddings, getTrueLabels, getTopics, loadPreprocessedText
from datetime import datetime
from xlrd import open_workbook
from xlutils.copy import copy
import xlwt
from sklearn.cluster import KMeans
import warnings
from labels.escenario_multilabel.performance import bad_pred_multilabel, ml_score
warnings.filterwarnings("ignore")

mode = multilabel_dict.split('_')[0]
label_set = list(dict.fromkeys(getTrueLabels(corpus)))

def get_centroid_predictions():
    preds = pd.read_csv('./results/%s/labels/multilabel/%s_centroid_multilabel_predicts_samples%s_percent%s.txt'%(model, mode, n_samples, percent), names=['pred'],header=None)
    preds['true'] = getTrueLabels(corpus)
    return preds

def get_centroid_discard_predictions():
    preds = pd.read_csv('./results/%s/labels/multilabel/%s_centroid_multilabel_predicts_samples%s_percent%s.txt'%(model, mode, n_samples, percent), names=['pred'],header=None)
    preds['true'] = getTrueLabels(corpus)
    preds['cluster'] = getTopics(model)
    preds = preds.loc[preds['cluster'] == -1]
    return preds

def evaluation(true, pred, title):
    tot = Counter(true)
    for label in label_set:
        if label not in tot.keys():
            tot.update({label : 0})
    evaluation = pd.DataFrame(index=label_set, columns=['precision', 'recall', 'f1'])
    evaluation['Total']=tot.values()
    cluster = getTopics(model)
    total = len(true)
    for label in label_set:
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        for i in range(len(true)):
            if label == true[i] and true[i] in pred[i]:
                true_positives += 1
                if cluster[i] in bad_pred_multilabel:
                    total += 1
            if label == true[i] and true[i] not in pred[i]:
                false_negatives += 1
            if label in pred[i] and true[i] not in pred[i]:
                false_positives += 1
            if label != true[i] and true[i] not in pred[i]:
                true_negatives += 1
                if cluster[i] in bad_pred_multilabel:
                    total += 1
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
    acc = i / total
    print('Acc: %s \t Prec: %s \t Recall: %s \t F1: %s'%(acc,prec,recall,f1))
    wb = xlwt.Workbook()
    w_sheet = wb.add_sheet('EVAL')
    w_sheet.write(0, 1, 'ACC', xlwt.easyxf('font: bold 1'))
    w_sheet.write(1, 1, 'PREC', xlwt.easyxf('font: bold 1'))
    w_sheet.write(2, 1, 'RECALL', xlwt.easyxf('font: bold 1'))
    w_sheet.write(3, 1, 'F1', xlwt.easyxf('font: bold 1'))
    w_sheet.write(0, 3, acc)
    w_sheet.write(1, 3, prec)
    w_sheet.write(2, 3, recall)
    w_sheet.write(3, 3, f1)
    wb.save('./results/%s/evaluation/%s.xls'%(model,title))


print('CENTROIDS MULTILABEL EVALUATION')
discards_pred = get_centroid_discard_predictions()
evaluation(discards_pred['true'].tolist(), discards_pred['pred'].tolist(), '%s_centroids_multilabel_evaluation_%s_%s'%(mode, n_samples, percent))
final_pred = get_centroid_predictions()
evaluation(final_pred['true'].tolist(), final_pred['pred'].tolist(), '%s_centroids_final_multilabel_evaluation_%s_%s'%(mode, n_samples, percent))

import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix
from config import model_label as model, corpus, percent, multilabel_dict, n_samples
from BERTclassifier import getTrueLabels, getTopics
from collections import Counter
import os.path
from subprocess import call
from xlrd import open_workbook
from xlutils.copy import copy
import xlwt
from labels.escenario_multilabel.performance import bad_pred_multilabel, ml_score

mode = multilabel_dict.split('_')[0]
label_set = list(dict.fromkeys(getTrueLabels(corpus)))

def get_multilabel_dict(multilabel_dict):
    return pd.read_csv('./labels/escenario_multilabel/multilabel_dicts/%s.csv'%(multilabel_dict), delimiter=',', names=['cluster', 'label'])


def label_samples(model, second_label_importance, multilabel_dict):
    clusters = getTopics(model)
    multilabel_dicts = get_multilabel_dict(multilabel_dict)
    multilabel_dicts = dict(zip(multilabel_dicts.cluster, multilabel_dicts.label))
    f = open('./results/%s/labels/multilabel/%s_multilabel_predictions_%s_%s.txt'%(model, mode, n_samples, second_label_importance), 'w', encoding='utf-8')
    for cluster in clusters:
        f.write(multilabel_dicts.get(cluster) + '\n')
    f.close()


def get_multilabel_predictions(model, second_label_importance):
    return list(pd.read_csv('./results/%s/labels/multilabel/%s_multilabel_predictions_%s_%s.txt'%(model, mode, n_samples, second_label_importance), header=None)[0].to_numpy())


def evaluate(model, second_label_importance):
    true = getTrueLabels(corpus)
    acc = accuracy_per_label(model, second_label_importance)
    pred = get_multilabel_predictions(model, second_label_importance)
    cluster = getTopics(model)
    evaluation = pd.DataFrame(index=label_set, columns=['accuracy', 'precision', 'recall', 'f1'])
    tot = sum(acc['Total'])
    print(bad_pred_multilabel)
    for label in label_set:
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        for i in range(len(true)):
            if label == true[i] and true[i] in pred[i]:
                true_positives += 1
                if cluster[i] in bad_pred_multilabel:
                   tot += 1
            if label == true[i] and true[i] not in pred[i]:
                false_negatives += 1
            if label in pred[i] and true[i] not in pred[i]:
                false_positives += 1
            if label != true[i] and true[i] not in pred[i]:
                true_negatives += 1
                if cluster[i] in bad_pred_multilabel:
                   tot += 1
        topic_precision = 0
        topic_recall = 0
        topic_f1 = 0
        topic_acc = acc.loc[label, 'Accuracy']
        if true_positives + false_positives != 0:
            topic_precision = true_positives / (true_positives + false_positives)
        if true_positives + false_negatives != 0:
            topic_recall = true_positives / (true_positives + false_negatives)
        if topic_precision + topic_recall != 0:
            topic_f1 = 2 * (topic_precision * topic_recall) / (topic_precision + topic_recall)
        evaluation.loc[label] = [topic_acc, topic_precision, topic_recall, topic_f1]
    evaluation['Total'] = acc['Total']
    final_numbers = []
    acc, prec, recall, f1 = (0,0,0,0)

    for eval in evaluation.iterrows():
        prec += (eval[1]['precision'] * eval[1]['Total'])/tot
        recall += (eval[1]['recall'] * eval[1]['Total'])/tot
        f1 += (eval[1]['f1'] * eval[1]['Total'])/tot

    j = 0
    for i in range(len(true)):
        if true[i] in pred[i]:
            j += 1
    acc = j/tot
    evaluation = evaluation.drop('Total', axis = 1)
    evaluation.loc['Total'] = [acc, prec, recall, f1]
    print(evaluation)

    wb = xlwt.Workbook()
    w_sheet = wb.add_sheet('EVAL')
    w_sheet = wb.get_sheet(0)
    w_sheet.write(0, 1, 'ACC', xlwt.easyxf('font: bold 1'))
    w_sheet.write(1, 1, 'PREC', xlwt.easyxf('font: bold 1'))
    w_sheet.write(2, 1, 'RECALL', xlwt.easyxf('font: bold 1'))
    w_sheet.write(3, 1, 'F1', xlwt.easyxf('font: bold 1'))
    w_sheet.write(0, 3, acc)
    w_sheet.write(1, 3, prec)
    w_sheet.write(2, 3, recall)
    w_sheet.write(3, 3, f1)
    wb.save('./results/%s/evaluation/%s_first_multilabel_evaluation_%s_%s.xls'%(model, mode, n_samples, percent))

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
         df.loc[label] = [discards_of_topic, '{:.2%}'.format(perc_discard), correct, total, acc, '{:.2%}'.format(pure_acc)]
    print(df.to_string())
    return df


label_samples(model, percent, multilabel_dict)
evaluate(model, percent)

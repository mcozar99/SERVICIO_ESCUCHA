import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import corpus, relabel, sentence_transformer, model_label as model, n_samples, percent
from sentence_transformers import SentenceTransformer
from BERTclassifier import getSamples, getTopics, loadPreprocessedText, getTrueLabels
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix
from datetime import datetime
from xlrd import open_workbook
from xlutils.copy import copy
import xlwt

mode = relabel.split('_')[0]
label_set = list(dict.fromkeys(getTrueLabels(corpus)))
topics = getTrueLabels(corpus)

def get_final_predicts():
    return pd.read_csv('./results/%s/labels/monolabel/%s_kneighbors_predicts_samples%s_percent%s.txt'%(model, mode, n_samples, percent), header=None, names = ['pred'])['pred'].tolist()

def get_discard_labels():
    df = pd.read_csv('./results/%s/labels/monolabel/%s_kneighbors_predicts_samples%s_percent%s.txt'%(model, mode, n_samples, percent), header=None, names = ['pred'])
    df['cluster'] = getTopics(model)
    return df[df['cluster'] == -1]['pred'].tolist()

def discards_evaluation():
    print('KNEIGHBORS EVALUATION', flush = True)
    y_pred = []
    y_true = []
    clusters = getTopics(model)
    final_predicts = get_final_predicts()
    for i in range(len(clusters)):
        if clusters[i] == -1:
            y_pred.append(final_predicts[i])
            y_true.append(topics[i])
    print(type(y_true), type(y_pred))
    print(y_true[0:10], y_pred[0:10])
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prec = precision_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    recall = recall_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    print('ACC: %s \nPREC: %s \nRECALL: %s \nF1: %s'%(acc, prec, recall, f1), flush=True)
    confusion_m = confusion_matrix(y_true, y_pred, label_set)
    print(pd.DataFrame(confusion_m, index=label_set,columns=label_set).to_string(), flush=True)
    matrix = pd.DataFrame(confusion_m, index=label_set,columns=label_set)
    matrix.to_excel('./results/%s/evaluation/%s_kneighbors_monolabel_evaluation_%s_%s.xls'%(model, mode, n_samples, percent), columns=label_set, index=label_set, startcol=1, startrow=1, merge_cells=True)
    rb = open_workbook('./results/%s/evaluation/%s_kneighbors_monolabel_evaluation_%s_%s.xls'%(model, mode, n_samples, percent))
    wb = copy(rb)
    w_sheet = wb.get_sheet(0)
    w_sheet.write(0, 0, model)
    w_sheet.write(len(label_set) + 2, 1, 'ACC', xlwt.easyxf('font: bold 1'))
    w_sheet.write(len(label_set) + 3, 1, 'PREC', xlwt.easyxf('font: bold 1'))
    w_sheet.write(len(label_set) + 4, 1, 'RECALL', xlwt.easyxf('font: bold 1'))
    w_sheet.write(len(label_set) + 5, 1, 'F1', xlwt.easyxf('font: bold 1'))
    w_sheet.write(len(label_set) + 2, 2, acc)
    w_sheet.write(len(label_set) + 3, 2, prec)
    w_sheet.write(len(label_set) + 4, 2, recall)
    w_sheet.write(len(label_set) + 5, 2, f1)
    wb.save('./results/%s/evaluation/%s_kneighbors_monolabel_evaluation_%s_%s.xls'%(model, mode, n_samples, percent))


def complete_evaluation():
    print('FINAL EVALUATION WITH KNEIGHBORS AND MONOLABELING', flush = True)
    y_true = getTrueLabels(corpus)
    y_pred = get_final_predicts()
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prec = precision_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    recall = recall_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    print('ACC: %s \nPREC: %s \nRECALL: %s \nF1: %s'%(acc, prec, recall, f1), flush=True)
    confusion_m = confusion_matrix(y_true, y_pred, label_set)
    print(pd.DataFrame(confusion_m, index=label_set,columns=label_set).to_string(), flush=True)
    matrix = pd.DataFrame(confusion_m, index=label_set,columns=label_set)
    matrix.to_excel('./results/%s/evaluation/%s_kneighbors_final_monolabel_evaluation_%s_%s.xls'%(model, mode, n_samples, percent), columns=label_set, index=label_set, startcol=1, startrow=1, merge_cells=True)
    rb = open_workbook('./results/%s/evaluation/%s_kneighbors_final_monolabel_evaluation_%s_%s.xls'%(model, mode, n_samples, percent))
    wb = copy(rb)
    w_sheet = wb.get_sheet(0)
    w_sheet.write(0, 0, model)
    w_sheet.write(len(label_set) + 2, 1, 'ACC', xlwt.easyxf('font: bold 1'))
    w_sheet.write(len(label_set) + 3, 1, 'PREC', xlwt.easyxf('font: bold 1'))
    w_sheet.write(len(label_set) + 4, 1, 'RECALL', xlwt.easyxf('font: bold 1'))
    w_sheet.write(len(label_set) + 5, 1, 'F1', xlwt.easyxf('font: bold 1'))
    w_sheet.write(len(label_set) + 2, 2, acc)
    w_sheet.write(len(label_set) + 3, 2, prec)
    w_sheet.write(len(label_set) + 4, 2, recall)
    w_sheet.write(len(label_set) + 5, 2, f1)
    wb.save('./results/%s/evaluation/%s_kneighbors_final_monolabel_evaluation_%s_%s.xls'%(model, mode, n_samples, percent))



discards_evaluation()
complete_evaluation()

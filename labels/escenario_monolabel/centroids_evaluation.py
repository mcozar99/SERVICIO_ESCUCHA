import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from config import model_label as model, corpus, relabel, n_samples, percent
from sentence_transformers import SentenceTransformer
from BERTclassifier import getEmbeddings, getTrueLabels, getTopics, loadPreprocessedText
from datetime import datetime
from xlrd import open_workbook
from xlutils.copy import copy
import xlwt
from sklearn.cluster import KMeans
import warnings
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix
from xlrd import open_workbook
from xlutils.copy import copy
import xlwt
warnings.filterwarnings("ignore")

label_set = list(dict.fromkeys(getTrueLabels(corpus)))
mode = relabel.split('_')[0]

def get_final_centroid_predicts():
    return pd.read_csv('./results/%s/labels/monolabel/%s_centroid_predicts_samples%s_percent%s.txt'%(model, mode, n_samples, percent), names=['pred'],header=None)['pred'].tolist()

def centroid_discards_evaluation(y_true, y_pred):
    print('CENTROID MODEL EVALUATION', flush = True)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prec = precision_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    recall = recall_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    print('ACC: %s \nPREC: %s \nRECALL: %s \nF1: %s'%(acc, prec, recall, f1))
    confusion_m = confusion_matrix(y_true, y_pred, label_set)
    print(pd.DataFrame(confusion_m, index=label_set,columns=label_set).to_string())
    matrix = pd.DataFrame(confusion_m, index=label_set,columns=label_set)
    matrix.to_excel('./results/%s/evaluation/%s_centroids_monolabel_evaluation_%s_%s.xls'%(model, mode, n_samples, percent), columns=label_set, index=label_set, startcol=1, startrow=1, merge_cells=True)
    rb = open_workbook('./results/%s/evaluation/%s_centroids_monolabel_evaluation_%s_%s.xls'%(model, mode, n_samples, percent))
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
    wb.save('./results/%s/evaluation/%s_centroids_monolabel_evaluation_%s_%s.xls'%(model, mode, n_samples, percent))

def centroid_complete_evaluation(y_true, y_pred):
    print('FINAL SYSTEM MONOLABELING+CENTORIDS EVALUATION', flush = True)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prec = precision_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    recall = recall_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    print('ACC: %s \nPREC: %s \nRECALL: %s \nF1: %s'%(acc, prec, recall, f1))
    confusion_m = confusion_matrix(y_true, y_pred, label_set)
    print(pd.DataFrame(confusion_m, index=label_set,columns=label_set).to_string())
    matrix = pd.DataFrame(confusion_m, index=label_set,columns=label_set)
    matrix.to_excel('./results/%s/evaluation/%s_centroids_final_monolabel_evaluation_%s_%s.xls'%(model, mode, n_samples, percent), columns=label_set, index=label_set, startcol=1, startrow=1, merge_cells=True)
    rb = open_workbook('./results/%s/evaluation/%s_centroids_final_monolabel_evaluation_%s_%s.xls'%(model, mode, n_samples, percent))
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
    wb.save('./results/%s/evaluation/%s_centroids_final_monolabel_evaluation_%s_%s.xls'%(model, mode, n_samples, percent))



tries = pd.DataFrame(get_final_centroid_predicts(), columns=['pred'])
tries['true'] = getTrueLabels(corpus)
tries['cluster'] = getTopics(model)

discards = tries[tries['cluster'] == -1]
print(discards)
centroid_discards_evaluation(discards['true'].tolist(), discards['pred'].tolist())
centroid_complete_evaluation(tries['true'].tolist(), tries['pred'].tolist())

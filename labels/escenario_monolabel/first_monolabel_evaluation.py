import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import pandas as pd
import numpy as np
from BERTclassifier import getTopics, loadPreprocessedText, getTrueLabels
from collections import Counter
import random
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix
import xlwt
from xlrd import open_workbook
from xlutils.copy import copy
from subprocess import call
from config import corpus, relabel, model_label as model, n_samples, percent
import os.path
r = lambda: random.randint(0,255) #Generador de numeros aleatorios para colores

label_set = list(dict.fromkeys(getTrueLabels(corpus)))

if not os.path.exists('./results/%s/evaluation'%model):
    call('mkdir ./results/%s/evaluation'%model, shell=True)

def reclassify(relabel, model):
    df = pd.read_csv('./labels/escenario_monolabel/label_dict/%s'%relabel, header=None, delimiter=',', names=['cluster', 'label'])
    df = dict(zip(df.cluster, df.label))
    label_list=[]
    for cluster in getTopics(model):
        label_list.append(df.get(cluster))
    return label_list

def accuracyXTopic(corpus, model, relabel):
    # HISTOGRAM OF PRECISION IN EACH TOPIC OF A MODEL
    clasification = reclassify(relabel, model)
    topics = getTrueLabels(corpus)
    topic_set = label_set
    accuracy_list = []
    df = pd.DataFrame(columns = ['Corrects', 'Total', 'Accuracy'], index=topic_set)
    for topic in topic_set:
        corrects = 0
        for i in range(len(clasification)):
            if topic in clasification[i] and clasification[i] in topics[i]:
                corrects+=1
        accuracy_list.append(corrects/Counter(topics)[topic])
        df.loc[topic] = [corrects, Counter(topics)[topic], '{:.2%}'.format(corrects/Counter(topics)[topic])]
    print('FIRST ACCURACY X TOPIC')
    print(df)

def evaluation(corpus, model, relabel):
    mode = relabel.split('_')[0]
    print('EVALUATING MODEL: %s'%model)
    y_pred = reclassify(relabel, model)
    y_true = getTrueLabels(corpus)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prec = precision_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    recall = recall_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    print('FIRST EVALUATION')
    print('ACC: %s \nPREC: %s \nRECALL: %s \nF1: %s'%(acc, prec, recall, f1))
    cm = confusion_matrix(y_true, y_pred, label_set)
    confusion_m = []
    for line in cm:
        confusion_m.append(line)
    ##################################### RESULTS TO EXCEL
    matrix = pd.DataFrame(confusion_m, index=label_set,columns=label_set)
    matrix.to_excel('./results/%s/evaluation/%s_first_monolabel_evaluation_%s_%s.xls'%(model, mode, n_samples, percent), columns=label_set, index=label_set, startcol=1, startrow=1, merge_cells=True)
    rb = open_workbook('./results/%s/evaluation/%s_first_monolabel_evaluation_%s_%s.xls'%(model, mode, n_samples, percent))
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
    wb.save('./results/%s/evaluation/%s_first_monolabel_evaluation_%s_%s.xls'%(model, mode, n_samples, percent))
    #################################### EDIT EXCEL
    print(matrix.to_string())



evaluation(corpus, model, relabel)
accuracyXTopic(corpus,model,relabel)

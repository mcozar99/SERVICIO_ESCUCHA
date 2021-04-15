import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from config import centroid_label, centroid_evaluation, model_label as model, corpus, relabel
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

def reclassify(relabel, model):
    df = pd.read_csv('./labels/escenario_monolabel/label_dict/%s'%relabel, header=None, delimiter=',', names=['cluster', 'label'])
    df = dict(zip(df.cluster, df.label))
    label_list=[]
    for cluster in getTopics(model):
        label_list.append(df.get(cluster))
    return label_list

df = pd.DataFrame(reclassify(relabel, model), columns = ['pred'])
df['true'] = getTrueLabels(corpus)
df['text'] = loadPreprocessedText(corpus)
df = df.assign(code=[*getEmbeddings(model, 'numpy')])
df['cluster'] = getTopics(model)
label_set = list(dict.fromkeys(getTrueLabels(corpus)))
cluster_set = list(dict.fromkeys(df['cluster']))
cluster_set.remove(-1)
print(df)

def k_means(points):
    #if len(points) < 1:
    #    return
    # Object KMeans
    kmeans = KMeans(n_clusters=1, max_iter=30,
                    init='k-means++', tol=0.001, n_jobs=8)
    # Calculate Kmeans
    kmeans.fit(points)
    # Obtain centroids and number Cluster of each point
    centroids = kmeans.cluster_centers_
    num_cluster_points = kmeans.labels_.tolist()
    return centroids[0]


def centroides(df):
    discards = df[df['pred'] == 'descarte']
    df.drop(df[df['pred'] == 'descarte'].index, inplace=True)
    centroids_dataset = pd.DataFrame(index=cluster_set, columns=['centroid'])
    for cluster in cluster_set:
        points = df[df['cluster'] == cluster]['code'].tolist()
        centroid = k_means(points)
        centroids_dataset.loc[cluster]['centroid'] = centroid
    print(centroids_dataset)
    for item in discards.iterrows():
        distance = cosine_similarity([item[1]['code']], centroids_dataset['centroid'].tolist())
        centroids_dataset['distance'] = distance[0]
        cluster_assigned = centroids_dataset[centroids_dataset.distance == centroids_dataset.distance.max()].index[0]
        item[1]['pred'] = df[df['cluster'] == cluster_assigned]['pred'].tolist()[0]
        discards.loc[item[0]] = item[1]
    frames = [df, discards]
    df = pd.concat(frames).sort_index(axis=0)
    df['pred'].to_csv('./results/%s/centroid_predicts.txt'%model, encoding='utf-8')
    return df

def get_final_centroid_predicts():
    final_predicts = []
    for line in open('./results/%s/labels/monolabel/centroid_predicts.txt'%model, 'r', encoding='utf-8'):
        final_predicts.append(line.replace('\n', ""))
    return final_predicts

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
    matrix.to_excel('./results/%s/evaluation/centroids_evaluation.xls'%model, columns=label_set, index=label_set, startcol=1, startrow=1, merge_cells=True)
    rb = open_workbook('./results/%s/evaluation/centroids_evaluation.xls'%model)
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
    wb.save('./results/%s/evaluation/centroids_evaluation.xls'%model)

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
    matrix.to_excel('./results/%s/evaluation/centroids_final_evaluation.xls'%model, columns=label_set, index=label_set, startcol=1, startrow=1, merge_cells=True)
    rb = open_workbook('./results/%s/evaluation/centroids_final_monolabel_evaluation.xls'%model)
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
    wb.save('./results/%s/evaluation/centroids_final_monolabel_evaluation.xls'%model)



if centroid_label:
    centroides(df)


if centroid_evaluation:
    tries = pd.DataFrame(get_final_centroid_predicts(), columns=['pred'])
    tries['true'] = getTrueLabels(corpus)
    tries['f_pred'] = df['pred']
    print(tries)
    discards = tries[tries['f_pred'] == 'descarte']
    centroid_discards_evaluation(discards['true'].tolist(), discards['pred'].tolist())
    centroid_complete_evaluation(tries['true'].tolist(), tries['pred'].tolist())

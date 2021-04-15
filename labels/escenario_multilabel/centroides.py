import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from config import corpus, model_label as model, percent, sentence_transformer, centroids_labeling, centroids_eval
from sentence_transformers import SentenceTransformer
from BERTclassifier import getEmbeddings, getTrueLabels, getTopics, loadPreprocessedText
from datetime import datetime
from xlrd import open_workbook
from xlutils.copy import copy
import xlwt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('results/%s/labels/multilabel/multilabel_predictions_%s.txt'%(model, percent), names=['pred'], header=None)
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
    df['pred'].to_csv('./results/%s/labels/multilabel/centroid_multilabel_predicts.txt'%model, index=False, header=False)

def get_centroid_predictions():
    preds = pd.read_csv('./results/%s/labels/multilabel/centroid_multilabel_predicts.txt'%model, names=['pred'],header=None)
    preds['true'] = getTrueLabels(corpus)
    return preds

def get_centroid_discard_predictions():
    preds = pd.read_csv('./results/%s/labels/multilabel/centroid_multilabel_predicts.txt'%model, names=['pred'],header=None)
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

if centroids_labeling:
    centroides(df)
if centroids_eval:
    print('CENTROIDS EVALUATION')
    print('DISCARDS EVALUATION')
    discards_pred = get_centroid_discard_predictions()
    evaluation(discards_pred['true'].tolist(), discards_pred['pred'].tolist(), 'centroids_multilabel_evaluation')
    print('FINAL EVALUATION')
    final_pred = get_centroid_predictions()
    evaluation(final_pred['true'].tolist(), final_pred['pred'].tolist(), 'centroids_final_multilabel_evaluation')

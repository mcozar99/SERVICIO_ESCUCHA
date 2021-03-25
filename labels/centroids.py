import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from BERTclassifier import getTopics
from config import centroid_label, centroid_evaluation, centroid_plot, model_label as model, corpus, relabel
from labels.model_similarity import accurate_embeddings_codification
from labels.common_functions import getTopicList, reclassify, get_label_set, sbert_model, predicts, get_accurate_embeddings_codification, get_accurate_indexes, text
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import random
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix
from xlrd import open_workbook
from xlutils.copy import copy
import xlwt

# -*- coding: utf-8 -*-
__author__ = 'Miguel Cozar'

r = lambda: random.randint(0,255) #Generador de numeros aleatorios para colores

# Constant
NUM_CLUSTERS = 1
MAX_ITERATIONS = 30
INITIALIZE_CLUSTERS = ['k-means++', 'random']
CONVERGENCE_TOLERANCE = 0.001
NUM_THREADS = 8

index_correct = get_accurate_indexes()
label_set = get_label_set(corpus)
topics = getTopicList(corpus)
accurate_embeddings_codification = get_accurate_embeddings_codification(corpus, label_set)

def reclassify(relabel, model):
    # PUTS LABELS TO EVERY CLUSTER DEPENDING ON THE DICTIONARY
    diccionario = relabelDict(relabel)
    nuevo = getTopics(model=model)
    new_topic_list = []
    for line in nuevo:
        new_topic_list.append(diccionario.get(line))
    print('PUT LABELS IN CLUSTERS')
    return new_topic_list

def getTopicList(corpus):
    # GETS ACCURATE LABELS FOR EVERY INPUT
    topics = []
    for line in open('./corpus/preprocessed/preprocess_%s'%corpus, 'r', encoding='utf-8'):
        topics.append(line.split('\t')[1].strip())
    return topics

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def k_means(embeddings, num_clusters, max_iterations, init_cluster, tolerance,
            num_threads):
    if len(embeddings) < 1:
        return
    points = embeddings
    # Object KMeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=max_iterations,
                    init=init_cluster, tol=tolerance, n_jobs=num_threads)
    # Calculate Kmeans
    kmeans.fit(points)
    # Obtain centroids and number Cluster of each point
    centroids = kmeans.cluster_centers_
    num_cluster_points = kmeans.labels_.tolist()
    return centroids

def determine_proximity_to_centroid(input, topic):
    input = [sbert_model.encode(input)]
    if centroids.get(topic) is None:
        return np.array([-1])
    return cosine_similarity(input, centroids.get(topic))

def label_discard_centroid(discard):
    proximities = []
    for label in label_set:
        proximities.append(determine_proximity_to_centroid(discard, label))
    #print(label_set[np.argmax(proximities)])
    return label_set[np.argmax(proximities)]


def plotCentroids(centroids):  #CENTROIDS IS THE DICTIONARY OF CENTROIDS
    color_list = []
    for label in label_set:
        color_list.append('#%02X%02X%02X' % (r(),r(),r()))
    values = list(centroids.values())
    j = 0
    aux_label_set=label_set
    while j < len(values):
        if values[j] is not None:
            values[j] = values[j].tolist()[0]
            j+=1
        else:
            values.pop(j)
            color_list.pop(j)
            aux_label_set.pop(j)
    if values.__len__() > 50:
        pca = PCA(n_components=50)
        values = pca.fit_transform(values)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, init="pca", n_iter=1000)
    tsne_results = tsne.fit_transform(values)
    j = 0
    for point in tsne_results:
        plt.plot(point[0], point[1], 'o',  color=color_list[j])
        plt.text(point[0], point[1], aux_label_set[j])
        j += 1
    plt.title('Centorid representation by label in TSNE model %s'%model)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('./labels/TSNE_centroid_representation/%s.png'%model)


def get_final_centroid_predicts():
    final_predicts = []
    for line in open('./results/%s/centroid_predicts.txt'%model, 'r', encoding='utf-8'):
        final_predicts.append(line.replace('\n', ""))
    return final_predicts


def centroid_discards_evaluation():
    print('CENTROID MODEL EVALUATION', flush = True)
    y_pred = []
    y_true = []
    clusters = getTopics(model)
    final_predicts = get_final_centroid_predicts()
    for i in range(len(clusters)):
        if clusters[i] == -1:
            y_pred.append(final_predicts[i])
            y_true.append(topics[i])
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prec = precision_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    recall = recall_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    print('ACC: %s \nPREC: %s \nRECALL: %s \nF1: %s'%(acc, prec, recall, f1))
    confusion_m = confusion_matrix(y_true, y_pred, label_set)
    print(pd.DataFrame(confusion_m, index=label_set,columns=label_set).to_string())
    matrix = pd.DataFrame(confusion_m, index=label_set,columns=label_set)
    matrix.to_excel('./results/%s/evaluation/kmeans_evaluation.xls'%model, columns=label_set, index=label_set, startcol=1, startrow=1, merge_cells=True)
    rb = open_workbook('./results/%s/evaluation/kmeans_evaluation.xls'%model)
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
    wb.save('./results/%s/evaluation/kmeans_evaluation.xls'%model)

def centroid_complete_evaluation():
    print('FINAL SYSTEM MONOLABELING+CENTORIDS EVALUATION', flush = True)
    y_true = topics
    y_pred = get_final_centroid_predicts()
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prec = precision_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    recall = recall_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    print('ACC: %s \nPREC: %s \nRECALL: %s \nF1: %s'%(acc, prec, recall, f1))
    confusion_m = confusion_matrix(y_true, y_pred, label_set)
    print(pd.DataFrame(confusion_m, index=label_set,columns=label_set).to_string())
    matrix = pd.DataFrame(confusion_m, index=label_set,columns=label_set)
    matrix.to_excel('./results/%s/evaluation/kmeans_final_evaluation.xls'%model, columns=label_set, index=label_set, startcol=1, startrow=1, merge_cells=True)
    rb = open_workbook('./results/%s/evaluation/kmeans_final_evaluation.xls'%model)
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
    wb.save('./results/%s/evaluation/kmeans_final_evaluation.xls'%model)



centroids = {}
for label in label_set:
    embeddings_set = accurate_embeddings_codification.get(label)
    if accurate_embeddings_codification.get(label) is not None:
        centroid = k_means(accurate_embeddings_codification.get(label), NUM_CLUSTERS, MAX_ITERATIONS, INITIALIZE_CLUSTERS[0], CONVERGENCE_TOLERANCE, NUM_THREADS)
    else:
        centroid = None
    centroids.update({label : centroid})
accurate_embeddings_codification = centroids
print(len(accurate_embeddings_codification))
print(len(accurate_embeddings_codification.values()))

if centroid_plot:
    plotCentroids(centroids)

if centroid_label:
    for i in range(len(predicts)):
        if 'desc' in predicts[i]:
            predicts[i] = label_discard_centroid(text[i])
        if i%7500 == 0:
            print(datetime.now().strftime('%H:%M:%S'))
            print(i)
    f = open('./results/%s/centroid_predicts.txt'%model, 'w', encoding='utf-8')
    for line in predicts:
        f.write(line.replace('\n', "") + '\n')
    f.close()

if centroid_evaluation:
    centroid_discards_evaluation()
    centroid_complete_evaluation()



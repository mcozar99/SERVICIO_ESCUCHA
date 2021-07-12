import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import pandas as pd
import numpy as np
from config import corpus, model_label as model, percent, sentence_transformer, kneighbors, centroids, multilabel_dict, n_samples, relabel
from BERTclassifier import getEmbeddings, getTrueLabels, getTopics, loadPreprocessedText
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from sklearn.cluster import KMeans
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")
label_set = list(dict.fromkeys(getTrueLabels(corpus)))

mode = multilabel_dict.split('_')[0]

print('LABELING MODEL %s WITH %s SAMPLES PER CLUSTER AND %s PERCENT'%(model, n_samples, percent))

def k_means(points):
    kmeans = KMeans(n_clusters=1, max_iter=30, init='k-means++', tol=0.001, n_jobs=8)
    kmeans.fit(points)
    centroids = kmeans.cluster_centers_
    num_cluster_points = kmeans.labels_.tolist()
    return centroids[0]

def reclassify(relabel, model):
    df = pd.read_csv('./labels/escenario_monolabel/label_dict/%s'%relabel, header=None, delimiter=',', names=['cluster', 'label'])
    df = dict(zip(df.cluster, df.label))
    label_list=[]
    for cluster in getTopics(model):
        label_list.append(df.get(cluster))
    return label_list

df = pd.read_csv('./results/%s/labels/multilabel/%s_multilabel_predictions_%s_%s.txt'%(model, mode, n_samples, percent), names=['multilabel_pred'], header=None)
df['monolabel_pred'] = reclassify(relabel, model)
df['true'] = getTrueLabels(corpus)
df['cluster'] = getTopics(model)
df = df.assign(code=[*getEmbeddings(model, 'numpy')])

cluster_set = list(dict.fromkeys(df['cluster']))
cluster_set.remove(-1)

if not os.path.isfile('./results/%s/labels/centroids_cluster_assignment.txt'%(model)):
    print('NOT DETECTED CENTROIDS CLUSTER ASSIGNMENT, PROCEEDING')
    if centroids:
        print('CENTROIDS LABELING')
        discards = df[df['monolabel_pred'] == 'descarte']
        df.drop(df[df['monolabel_pred'] == 'descarte'].index, inplace=True)
        centroids_dataset = pd.DataFrame(index=cluster_set, columns=['centroid'])
        for cluster in cluster_set:
            points = df[df['cluster'] == cluster]['code'].tolist()
            centroid = k_means(points)
            centroids_dataset.loc[cluster]['centroid'] = centroid
        print(centroids_dataset, flush = True)
        i = 0
        for item in discards.iterrows():
            distance = cosine_similarity([item[1]['code']], centroids_dataset['centroid'].tolist())
            centroids_dataset['distance'] = distance[0]
            cluster_assigned = centroids_dataset[centroids_dataset.distance == centroids_dataset.distance.max()].index[0]
            discards.loc[item[0], 'monolabel_pred'] = df[df['cluster'] == cluster_assigned]['monolabel_pred'].tolist()[0]
            discards.loc[item[0], 'multilabel_pred'] = df[df['cluster'] == cluster_assigned]['multilabel_pred'].tolist()[0]
            discards.loc[item[0], 'cluster'] = df[df['cluster'] == cluster_assigned]['cluster'].tolist()[0]
            i += 1
            if i%10000 == 0:
                print(datetime.now().strftime('%H:%M:%S'), i, flush=True)

        frames = [df, discards]
        df = pd.concat(frames).sort_index(axis=0)
        print(df, flush=True)
        print('CENTROIDS LABELING COMPLETED, SAVING')
        df['multilabel_pred'].to_csv('./results/%s/labels/multilabel/%s_centroid_multilabel_predicts_samples%s_percent%s.txt'%(model, mode, n_samples, percent), index=False, header=False)
        df['monolabel_pred'].to_csv('./results/%s/labels/monolabel/%s_centroid_predicts_samples%s_percent%s.txt'%(model, mode, n_samples, percent), index=False, header=False)
        df['cluster'].to_csv('./results/%s/labels/centroids_cluster_assignment.txt'%(model), index=False, header=False)
        df['monolabel_pred'] = reclassify(relabel, model)
        df['multilabel_pred'] = pd.read_csv('./results/%s/labels/multilabel/%s_multilabel_predictions_%s_%s.txt'%(model, mode, n_samples, percent), header=None)
        df['cluster'] = getTopics(model)
else:
    print('DETECTED CENTROIDS CLUSTER ASSIGNMENT', flush = True)
    if centroids:
        df['cluster_assigned'] = pd.read_csv('./results/%s/labels/centroids_cluster_assignment.txt'%(model), header=None)
        i = 0
        for item in df.iterrows():
            if df.loc[item[0], 'cluster'] == -1:
                monolabel_pred = df[df['cluster'] == item[1].cluster_assigned].sample(1).monolabel_pred.item()
                multilabel_pred = df[df['cluster'] == item[1].cluster_assigned].sample(1).multilabel_pred.item()
                df.loc[item[0], 'monolabel_pred'] = monolabel_pred
                df.loc[item[0], 'multilabel_pred'] = multilabel_pred
                i += 1
                if i%10000 == 0:
                    print(datetime.now().strftime('%H:%M:%S'), i, flush=True)
        df['multilabel_pred'].to_csv('./results/%s/labels/multilabel/%s_centroid_multilabel_predicts_samples%s_percent%s.txt'%(model, mode, n_samples, percent), index=False, header=False)
        df['monolabel_pred'].to_csv('./results/%s/labels/monolabel/%s_centroid_predicts_samples%s_percent%s.txt'%(model, mode, n_samples, percent), index=False, header=False)
        print(df)
        df['monolabel_pred'] = reclassify(relabel, model)
        df['multilabel_pred'] = pd.read_csv('./results/%s/labels/multilabel/%s_multilabel_predictions_%s_%s.txt'%(model, mode, n_samples, percent), header=None)


if not os.path.isfile('./results/%s/labels/kneighbors_cluster_assignment.txt'%(model)):
    print('NOT DETECTED KNN CLUSTER ASSIGNMENT, PROCEEDING')
    if kneighbors:
        print('KNEIGHBORS LABELING')
        discards = df[df['monolabel_pred'] == 'descarte']
        df.drop(df[df['monolabel_pred'] == 'descarte'].index, inplace=True)
        i = 0
        for item in discards.iterrows():
            distance = cosine_similarity([item[1]['code']], df['code'].tolist())
            df['distance'] = distance[0]

            largest = df.nlargest(5, 'distance')
            monolabel= max(Counter(largest.monolabel_pred), key = Counter(largest.monolabel_pred).get)
            multilabel= max(Counter(largest.multilabel_pred), key = Counter(largest.multilabel_pred).get)
            cluster_assigned = max(Counter(largest.cluster), key = Counter(largest.cluster).get)

            discards.loc[item[0], 'monolabel_pred'] = monolabel
            discards.loc[item[0], 'multilabel_pred'] = multilabel
            discards.loc[item[0], 'cluster'] = df[df['cluster'] == cluster_assigned]['cluster'].tolist()[0]
            i += 1
            if i%5000 == 0:
                print(datetime.now().strftime('%H:%M:%S'), i, flush = True)
        frames = [df, discards]
        df = pd.concat(frames).sort_index(axis=0)
        print('FINISHED KNEIGHBORS, LABELING')
        #df['monolabel_pred'].to_csv('./results/%s/labels/multilabel/kneighbors_multilabel_predicts.txt'%model, index=False, header=False)
        df['multilabel_pred'].to_csv('./results/%s/labels/multilabel/%s_kneighbors_multilabel_predicts_samples%s_percent%s.txt'%(model, mode, n_samples, percent), index=False, header=False)
        df['monolabel_pred'].to_csv('./results/%s/labels/monolabel/%s_kneighbors_predicts_samples%s_percent%s.txt'%(model, mode, n_samples, percent), index=False, header=False)
        df['cluster'].to_csv('./results/%s/labels/kneighbors_cluster_assignment.txt'%(model), index=False, header=False)

else:
    print('DETECTED KNN CLUSTER ASSIGNMENT')
    if kneighbors:
        df['cluster_assigned'] = pd.read_csv('./results/%s/labels/kneighbors_cluster_assignment.txt'%(model), header=None)
        i = 0
        for item in df.iterrows():
            if df.loc[item[0], 'cluster'] == -1:
                monolabel_pred = df[df['cluster'] == item[1].cluster_assigned].sample(1).monolabel_pred.item()
                multilabel_pred = df[df['cluster'] == item[1].cluster_assigned].sample(1).multilabel_pred.item()
                df.loc[item[0], 'monolabel_pred'] = monolabel_pred
                df.loc[item[0], 'multilabel_pred'] = multilabel_pred
                i += 1
                if i%10000 == 0:
                    print(datetime.now().strftime('%H:%M:%S'), i, flush=True)
        print(df)
        df['multilabel_pred'].to_csv('./results/%s/labels/multilabel/%s_kneighbors_multilabel_predicts_samples%s_percent%s.txt'%(model, mode, n_samples, percent), index=False, header=False)
        df['monolabel_pred'].to_csv('./results/%s/labels/monolabel/%s_kneighbors_predicts_samples%s_percent%s.txt'%(model, mode, n_samples, percent), index=False, header=False)

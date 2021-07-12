import pandas as pd
import hdbscan
import numpy as np
import random
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from BERTclassifier import getTopics, getTrueLabels, getEmbeddings
from config import corpus, model_label as model, multilabel_dict, n_samples, percent
from sklearn.cluster import KMeans

def k_means(points):
    kmeans = KMeans(n_clusters=1, max_iter=30,
                    init='k-means++', tol=0.001, n_jobs=8)
    kmeans.fit(points)
    centroids = kmeans.cluster_centers_
    num_cluster_points = kmeans.labels_.tolist()
    return centroids[0]

mode = multilabel_dict.split('_')[0]
label_set = list(dict.fromkeys(getTrueLabels(corpus)))
cluster_set = list(dict.fromkeys(getTopics(model)))

dictionary = pd.read_csv('./labels/escenario_multilabel/multilabel_dicts/%s.csv'%multilabel_dict, header=None, names=['cluster', 'label'])
labels_centroids = pd.DataFrame(index=label_set, columns=['centroid'])
clusters_centroids = pd.DataFrame(index=cluster_set, columns=['centroid'])


df = pd.DataFrame(getTrueLabels(corpus), columns=['true'])
df['cluster'] = getTopics(model)
df = df.assign(code=[*getEmbeddings(model, 'numpy')])

for label in label_set:
        points = df[df['true'] == label]['code'].tolist()
        centroid = k_means(points)
        labels_centroids.loc[label]['centroid'] = centroid

for cluster in cluster_set:
        points = df[df['cluster'] == cluster]['code'].tolist()
        centroid = k_means(points)
        clusters_centroids.loc[cluster]['centroid'] = centroid


pca = PCA(n_components=2)

labels_centroids['x'], labels_centroids['y'] = (pca.fit_transform(labels_centroids['centroid'].tolist())[:, 0], pca.fit_transform(labels_centroids['centroid'].tolist())[:, 1])
clusters_centroids['x'], clusters_centroids['y'] = (pca.fit_transform(clusters_centroids['centroid'].tolist())[:, 0], pca.fit_transform(clusters_centroids['centroid'].tolist())[:, 1])
clusters_centroids['label'] = dictionary['label']

plt.figure(figsize=(25, 9))
sns.scatterplot(
    x = 'x', y = 'y',
    palette= sns.color_palette("hls", n_colors=labels_centroids.shape[0]),
    data=labels_centroids
)



for i in range(labels_centroids.shape[0]):
    plt.text(x=labels_centroids.loc[list(labels_centroids.index)[i], 'x'] + 0.001,
             y=labels_centroids.loc[list(labels_centroids.index)[i], 'y'] + 0.001,
             s=list(labels_centroids.index)[i], fontdict=dict(color='black',size=8))

n_colors = list(dict.fromkeys(clusters_centroids['label'].tolist())).__len__()

sns.scatterplot(
    x = 'x', y = 'y',
    data=clusters_centroids,
    hue = clusters_centroids['label']
)

"""
for i in range(clusters_centroids.shape[0]):
    if list(clusters_centroids.index)[i] == -1:
        continue
    text = str(list(clusters_centroids.index)[i]) + ',' + dictionary.loc[list(clusters_centroids.index)[i], 'label']
    plt.text(x=clusters_centroids.loc[list(clusters_centroids.index)[i], 'x'] + 0.001,
             y=clusters_centroids.loc[list(clusters_centroids.index)[i], 'y'] + 0.001,
             s=text, fontdict=dict(color='black',size=8))
"""

plt.savefig('pruebas.png')
plt.show()

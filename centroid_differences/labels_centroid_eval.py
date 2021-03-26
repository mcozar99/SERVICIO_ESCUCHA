import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from BERTclassifier import loadPreprocessedText, getTrueLabels, getEmbeddings
from config import corpus, model_label as model, sentence_transformer, plot_centroid_distances, calculate_centroid_distances
from subprocess import call
import os.path


true_labels = getTrueLabels(corpus)
predictions = []
for line in open('./results/%s/predicts.txt'%model, 'r', encoding='utf-8'):
    predictions.append(line.replace('\n','').strip())
embeddings_code = getEmbeddings(model, 'numpy')
label_set = list(dict.fromkeys(true_labels))


def get_topic_centroid(embeddings):
    kmeans = KMeans(n_clusters=1, max_iter=100, init='k-means++', tol=0.001, n_jobs=8)
    kmeans.fit(embeddings)
    centroid = kmeans.cluster_centers_
    return centroid[0]

def get_organised_labels(labels):
    organised_dictionary = {}
    for label in label_set:
        embeddings = []
        indexes = []
        for i in range(len(labels)):
            if label in labels[i]:
                indexes.append(i)
        for index in indexes:
            embeddings.append(embeddings_code[index])
        if embeddings == []:
            organised_dictionary.update({label : None})
            continue
        organised_dictionary.update({label : embeddings})
    return organised_dictionary


def get_true_centroids(true_organised_labels):
    true_centroid_dict = {}
    for label in label_set:
        centroid = get_topic_centroid(true_organised_labels.get(label))
        true_centroid_dict.update({('true_%s'%label) : centroid})
    return true_centroid_dict


def get_pred_centroids(pred_organised_labels):
    pred_centroid_dict = {}
    for label in label_set:
        if pred_organised_labels.get(label) is None:
            pred_centroid_dict.update({('pred_%s'%label) : None})
            continue
        centroid = get_topic_centroid(pred_organised_labels.get(label))
        pred_centroid_dict.update({('pred_%s'%label) : centroid})
    return pred_centroid_dict



def get_centroid_data(pred, true):
    pred = {k: v for k, v in pred.items() if v is not None}
    pca = PCA(n_components = 2)
    true_centroids = pca.fit_transform(list(true.values()))
    pred_centroids = pca.fit_transform(list(pred.values()))

    pred = pd.DataFrame(dict(zip(list(pred.keys()), pred_centroids)).items(), columns= ['Label', 'Centroid'])
    true = pd.DataFrame(dict(zip(list(true.keys()), true_centroids)).items(), columns= ['Label', 'Centroid'])
    data = pred.append(true)
    style = np.append(np.full(len(pred), 'pred'), np.full(len(true), 'true'))

    return data, style


def plot_results(data, style):
    points = pd.DataFrame(list(data['Centroid']), columns=['x', 'y'])
    print(data)
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x='x', y='y',
        data=points,
        hue=style,
        style = style
    )

    for i in range(points.shape[0]):
        plt.text(x=list(points['x'])[i]+0.05, y=list(points['y'])[i]+0.05, s=list(data['Label'])[i], fontdict=dict(color='black',size=8))
        #, bbox=dict(facecolor='white',alpha=0.5))
    if not os.path.exists('./centroid_differences/representations'):
        call('mkdir ./centroid_differences/representations', shell=True)
    plt.title('%s Centroids Representation'%model)
    plt.savefig('./centroid_differences/representations/%s_centroids.png'%model)
    print('SAVED IN centroid_differences/representations/%s_centroids.png'%model)
    plt.show()


def get_centroid_distances(pred_centroids, true_centroids):
    labels = []
    distances = []
    similarities = []
    for label in label_set:
        centroids = []
        centroids.append(true_centroids.get('true_%s'%label))
        if pred_centroids.get('pred_%s'%label) is None:
            labels.append(label)
            distances.append(None)
            similarities.append(None)
            continue
        else:
            centroids.append(pred_centroids.get('pred_%s'%label))
            distances.append(euclidean_distances(centroids)[0,1])
            similarities.append(cosine_similarity(centroids)[0,1])
    df = pd.DataFrame(list(zip(distances, similarities)), index=label_set, columns=['euclidean', 'cosine'])
    print(df)

# DICTIONARY WITH KEYS = LABELS : VALUES = SAMPLES WITH LABEL IN KEY
pred_organised_labels = get_organised_labels(predictions)
true_organised_labels = get_organised_labels(true_labels)
# DICTIONARY WITH LABEL : CENTROID
pred = get_pred_centroids(pred_organised_labels)
true = get_true_centroids(true_organised_labels)

if plot_centroid_distances:
    print('PLOTTING CENTROIDS')
    data, style = get_centroid_data(pred, true)
    plot_results(data, style)
if calculate_centroid_distances:
    get_centroid_distances(pred, true)


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

def get_topic_centroid(embeddings):
    if embeddings.__len__() < 1:
        return np.zeros(512)
    kmeans = KMeans(n_clusters=1, max_iter=1500, init='k-means++', tol=0.0001, n_jobs=8)
    kmeans.fit(embeddings)
    centroid = kmeans.cluster_centers_
    return centroid[0]

fichero_predicciones = './results/%s/labels/monolabel/random_kneighbors_predicts_samples100000_percent40.txt'%model
predicciones_reales = getTrueLabels(corpus)

embeddings_code = getEmbeddings(model, 'numpy')

label_set = list(dict.fromkeys(predicciones_reales))
# ARRAY DE ETIQUETAS
etiquetas = []
for label in label_set:
    etiquetas.append('true_%s'%label)
    etiquetas.append('pred_%s'%label)

# DATAFRAME CON PREDICCIONES Y ETIQUETAS REALES, CON LOS EMBEDDINGS
df = pd.DataFrame(predicciones_reales, columns=['true'])
df['pred'] = pd.read_csv(fichero_predicciones, header=None)
df = df.assign(code=[*getEmbeddings(model, 'numpy')])

# DATAFRAME CON LOS CENTROIDES
centroids = pd.DataFrame(index = etiquetas, columns=['x', 'y'])

# CALCULO DE CADA UNO DE LOS CENTROIDES, REDUCCION DE DIMENSIONANLIDAD Y ASIGNACION
pca = PCA(n_components=2)
for label in label_set:
    print(label)
    # CENTROIDE
    centroide_true = get_topic_centroid(df[df.true.str.contains(label)].code.values.tolist())
    centroide_pred = get_topic_centroid(df[df.pred.str.contains(label)].code.values.tolist())
    # REDUCCION DIMENSIONAL
    centroide_true, centroide_pred = pca.fit_transform([centroide_true, centroide_pred])
    # ASIGNACION
    centroids.loc['true_%s'%label] = centroide_true
    centroids.loc['pred_%s'%label] = centroide_pred

print(centroids)

def plot_results(data, style):
    # Pasamos a DATAFRAME DE PUNTOS
    points = pd.DataFrame(list(data['centroide']), index = etiquetas, columns=['x', 'y'])
    print(data)
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x='x', y='y',
        data=points,
        hue=style,
        style = style
    )
    for i in range(points.shape[0]):
        plt.text(x=list(points['x'])[i] + 0.01, y=list(points['y'])[i] + 0.01, s=list(data['Label'])[i], fontdict=dict(color='black',size=8))
    # SALVAMOS Y CREAMOS CARPETA SI NO EXISTE
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
    return df


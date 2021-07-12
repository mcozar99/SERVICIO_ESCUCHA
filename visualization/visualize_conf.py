import sys
import numpy as np
import os.path
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import visualization.visualization as v
from BERTclassifier import getProbabilities, getEmbeddings
from config import model, formato, negros, modo, dimensions, represent, n_neighbors, mode


if 'probs' in formato:
    df = getProbabilities(model=model, format='df')
    embeddings = np.array(df)
    print('LOADED PROBS')
if 'embeddings' in formato:
    df = getEmbeddings(model=model, format='numpy')
    embeddings = np.array(df)
    print('LOADED EMBEDDINGS')


if 'tsne' in represent:
    if not os.path.isfile('./visualization/TSNE/tsne_result_%s_negros_%s.csv'%(model, negros)):
        print('NOT DETECTED MODEL, CREATING')
        if dimensions == 2:
            v.tsne(model=model, negros=negros, embeddings=embeddings, modo=modo)
        else:
            print('Only 2D representation for TSNE')
    else:
        print('DETECTED MODEL IN CACHE, LOADING AND PLOTTING')
        v.loadTSNE(file='tsne_result_%s_negros_%s'%(model, negros), model=model, embeddings=embeddings, negros=negros, title='tsne_%s'%(model), modo=modo)

if 'pca' in represent:
    if dimensions == 2:
        v.pca2D(model=model, embeddings=embeddings, negros=negros, modo=modo)
    else:
        v.pca(model=model, df=df, embeddings=embeddings, rndperm=rndperm, negros=negros)

if 'umap' in represent:
    if dimensions == 2:
        if mode == 'visualize':
            file = './visualization/UMAP_reducedDimensionality/%s_%sD_%sneighbors.csv'%(model, dimensions, n_neighbors)
            v.processUMAP(file=file, model=model, embeddings=embeddings, negros=negros, modo=modo)
        elif mode == 'train':
            v.uMAP(model=model, embeddings=embeddings, n_neighbors=n_neighbors, n_components=dimensions)
    else:
        if mode == 'visualize':
            file = './visualization/UMAP_reducedDimensionality/%s_%sD_%sneighbors.csv' % (model, dimensions, n_neighbors)
            v.umapping(model=model, negros=negros, embeddings=embeddings, df=df, rndperm=rndperm, file=file)
        elif mode == 'train':
            v.uMAP(model=model, embeddings=embeddings, n_neighbors=n_neighbors, n_components=dimensions)

if 'intertopic' in represent:
    v.visualizeModel(model)

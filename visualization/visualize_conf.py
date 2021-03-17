import sys
import numpy as np
import os.path
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import visualization as v
from BERTclassifier import getProbabilities

# TO USE VISUALIZATION.PY CONFIG THIS FILE AN EXEC IT FROM THE ROOT FOLDER /BERTOPIC
# python visualization/visualize_conf.py

###########################################################################################
# CONFIGURACION A
# TOCAR PARA VISUALIZAR
model = 'SENTENCE_TRANSFORMER_KNEIGHBORS_MINTOPICSIZE_30'             # MODELO A VISUALIZAR
negros = True               # TRUE PARA ENSEÑAR LOS TOPICS A -1 Y FALSE PARA OCULTARLOS
modo = ''              # topic muestra los nombres de los temas y texto los valores mas significativos, vacio para no mostrar nada
dimensions = 2              # 2 O 3 DIMENSIONES, RECOMENDADO 2
represent = ['tsne']        # INTERTOPIC, PCA UMAP O TSNE, PODEMOS METER VARIOS EN UNA LISTA
n_neighbors = 250          #SOLO PARA UMAP
mode = 'visualize'              # TRAIN PARA GENERAR MODELO, VISUALIZE PARA ENSEÑARLO, SOLO UMAP
###########################################################################################

if 'tsne' in represent:
    if not os.path.isfile('./visualization/TSNE/tsne_result_%s_negros_%s.csv'%(model, negros)):
        print('NOT DETECTED MODEL, CREATING')
        if dimensions == 2:
            v.tsne(model=model, negros=negros, embeddings=embeddings, modo=modo)
        else:
            print('Only 2D representation for TSNE')
    else:
        print('DETECTED MODEL IN CACHE, LOADING AND PLOTTING')
        v.loadTSNE(file='tsne_result_%s_negros_%s'%(model, negros), model=model, embeddings=embeddings, negros=negros, title='tsne_%s_negros_%s'%(model, negros), modo=modo)

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
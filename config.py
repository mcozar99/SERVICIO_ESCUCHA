import sys

# FICHERO DE CONFIGURACION DONDE SE MUEVEN TODOS LOS PARAMETROS


actions = ['t1rain', 'i1nfo', 's1ilhouette', 's1tats', 'v1isualization', 'l1abel', 'c1entroid_evaluation', 'test']         # SILHOUETTE, INFO, TRAIN, STATS, VISUALIZATION, MONOLABEL_EVALUATION, MULTILABEL_EVALUATION, CENTROID_EVALUATION


#GENERAL CONFIG
model_name = 'TFG'
min_topic_sizes = [30]
sentence_transformer = 'distiluse-base-multilingual-cased-v2' #"distiluse-base-multilingual-cased-v2" "dccuchile/bert-base-spanish-wwm-cased" 'bert-large-nli-mean-tokens'
corpus_pie = False

# TRAIN CONFIG
corpus = 'TRAIN_CORPUS_SERVICIO_ESCUCHA.txt'
iterations = 5                                #Number of tries to improve %classified


# INFO CONFIG
info = ['t1opics', 'info']          # TOPICS Y/O INFO
n_topics = 1000            # Number of topics to show


# SILHOUETTE CONFIG
metric = 'cosine'       #metric to measure results: euclidean, cosine, manhattan, chebyshev...
save = True             #TRUE TO SAVE FIGURE IN SILHOUETTE/SILHOUETTES

# STATS CONFIG
# USE MODEL_LABEL VARIABLE TO SELECT MODEL TO EXTRACT STATS
model_stats = True     # STATS OF MODEL CLASSIFICATIONS AND DISCARDS
label_stats = False      # STATS OF LABELS OF A CORPUS


# VISUALIZATION CONFIG
model = 'TFG_MINTOPICSIZE_30'             # MODELO A VISUALIZAR
formato = 'probs'      #REPRESENTAMOS EMBEDDINGS O PROBS
negros = False               # TRUE PARA ENSEÑAR LOS TOPICS A -1 Y FALSE PARA OCULTARLOS
modo = ''              # topic muestra los nombres de los temas y texto los valores mas significativos, vacio para no mostrar nada
dimensions = 2              # 2 O 3 DIMENSIONES, RECOMENDADO 2
represent = ['intertopic']        # INTERTOPIC, PCA UMAP O TSNE, PODEMOS METER VARIOS EN UNA LISTA
n_neighbors = 250           #SOLO PARA UMAP
mode = 'visualize'          # TRAIN PARA GENERAR MODELO, VISUALIZE PARA ENSEÑARLO, SOLO UMAP



## CENTROID DIFFERENCE EVALUATION USE VARIABLE MODEL_LABEL TO SELECT YOUR MODEL TO EVALAUTE
plot_centroid_distances =True         # PLOTS PREDICTIONS AND REAL CENTROIDS PER LABEL
calculate_centroid_distances = True    # DISPLAYS DISTANCES BETWEEN TRUE AND PRED CENTROIDS


# PARA LOS NUEVOS DICCIONARIOS
model_label = 'TFG_MINTOPICSIZE_30'                    # MODEL TO RELABEL
mode = 'random'                                         # RANDOM O PROBABILITIES
n_samples = int(sys.argv[1])
percent = int(sys.argv[2])   					# SECOND LABEL IMPORTANCE PERCENT
probabilistic_mode = True
random_mode = True
multilabel_dict = '%s_multilabel_%s_%s_%s'%(mode, model_label, n_samples, percent)  # PROBABILITIES O RANDOM + MODELO + PERCENT
relabel = '%s_dict_%s_%s_%s.txt'%(mode, model_label, n_samples, percent) # RELABEL DICTIONARY, CHEK LABELS/LABEL_DICT ONCE GENERATED
kneighbors = True
centroids = True

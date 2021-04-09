# FICHERO DE CONFIGURACION DONDE SE MUEVEN TODOS LOS PARAMETROS


actions = ['t1rain', 'i1nfo', 's1iluette', 's1tats', 'v1isualization', 'monolabel_evaluation', 'multilabel_evaluation', 'c1entroid_evaluation']         # SILHOUETTE, INFO, TRAIN, STATS, VISUALIZATION, MONOLABEL_EVALUATION, MULTILABEL_EVALUATION, CENTROID_EVALUATION


# GENERAL CONFIG
model_name = 'TEST'
min_topic_sizes = [1500]
sentence_transformer = "dccuchile/bert-base-spanish-wwm-cased"
corpus_pie = False

# TRAIN CONFIG
corpus = 'CORPUS_SERVICIO_ESCUCHA.txt'
iterations = 5                                #Number of tries to improve %classified


# INFO CONFIG
info = ['topics', 'info']          # TOPICS Y/O INFO
n_topics = 60            # Number of topics to show


# SILHOUETTE CONFIG
metric = 'cosine'       #metric to measure results: euclidean, cosine, manhattan, chebyshev...
save = True             #TRUE TO SAVE FIGURE IN SILHOUETTE/SILHOUETTES

# STATS CONFIG
# USE MODEL_LABEL VARIABLE TO SELECT MODEL TO EXTRACT STATS
model_stats = False     # STATS OF MODEL CLASSIFICATIONS AND DISCARDS
label_stats = True      # STATS OF LABELS OF A CORPUS


# VISUALIZATION CONFIG
model = 'NEW_PREPROCESS_CHILE_MINTOPICSIZE_100'             # MODELO A VISUALIZAR
formato = 'embeddings'      #REPRESENTAMOS EMBEDDINGS O PROBABILITIES
negros = True               # TRUE PARA ENSEÑAR LOS TOPICS A -1 Y FALSE PARA OCULTARLOS
modo = 'topic'              # topic muestra los nombres de los temas y texto los valores mas significativos, vacio para no mostrar nada
dimensions = 2              # 2 O 3 DIMENSIONES, RECOMENDADO 2
represent = ['pca']        # INTERTOPIC, PCA UMAP O TSNE, PODEMOS METER VARIOS EN UNA LISTA
n_neighbors = 250           #SOLO PARA UMAP
mode = 'visualize'          # TRAIN PARA GENERAR MODELO, VISUALIZE PARA ENSEÑARLO, SOLO UMAP


# LABELING CONFIG
# MONOLABEL
mono_label_number = 4            # 1 ONLY FIRST LABELING, 2 LABEL +  KNEIGHBORS, 3 LABEL + CENTROIDS, 4 LABEL + KNEIGBORS AND LABEL + CENTROIDS
model_label = 'TEST_MINTOPICSIZE_1500'                    # MODEL TO RELABEL
relabel = 'optimus_dictionary_TEST_MINTOPICSIZE_1500.txt' # RELABEL DICTIONARY, CHEK LABELS/LABEL_DICT ONCE GENERATED
##### FIRST EVALUATION
label_importance_in_cluster = False
global optimus_dictionary
global effective_dictionary
optimus_dictionary = True
effective_dictionary = False
evaluacion = True
##### KNEIGHBORS
eval_model_similarity = True          #EVALUATING
label_in_model_similarity = True      #LABELING
##### CENTROIDS
centroid_label = True
centroid_evaluation = True
centroid_plot = False


# MULTILABEL (SAME MODEL AS MONOLABEL)
multi_label_number = 1  # 1 ONLY FIRST LABELING, 2 MULTI-LABEL +  KNEIGHBORS, 3 LABEL + CENTROIDS, 4 MULTI-LABEL + KNEIGBORS AND LABEL + CENTROIDS
percent = 30    # SECOND LABEL IMPORTANCE PERCENT


## CENTROID DIFFERENCE EVALUATION USE VARIABLE MODEL_LABEL TO SELECT YOUR MODEL TO EVALAUTE
plot_centroid_distances = False         # PLOTS PREDICTIONS AND REAL CENTROIDS PER LABEL
calculate_centroid_distances = True    # DISPLAYS DISTANCES BETWEEN TRUE AND PRED CENTROIDS

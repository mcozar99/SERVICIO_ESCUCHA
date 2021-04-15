# FICHERO DE CONFIGURACION DONDE SE MUEVEN TODOS LOS PARAMETROS


actions = ['t1rain', 'i1nfo', 's1ilhouette', 's1tats', 'v1isualization', 'monolabel_evaluation', 'multilabel_evaluation', 'c1entroid_evaluation']         # SILHOUETTE, INFO, TRAIN, STATS, VISUALIZATION, MONOLABEL_EVALUATION, MULTILABEL_EVALUATION, CENTROID_EVALUATION


# GENERAL CONFIG
model_name = 'ST2_MPDI'
min_topic_sizes = [30]
sentence_transformer = 'distiluse-base-multilingual-cased-v2' #"distiluse-base-multilingual-cased-v2" "dccuchile/bert-base-spanish-wwm-cased" 'bert-large-nli-mean-tokens'
corpus_pie = False

# TRAIN CONFIG
corpus = 'CORPUS_SERVICIO_ESCUCHA.txt'
iterations = 3                                #Number of tries to improve %classified


# INFO CONFIG
info = ['topics', 'info']          # TOPICS Y/O INFO
n_topics = 10            # Number of topics to show


# SILHOUETTE CONFIG
metric = 'cosine'       #metric to measure results: euclidean, cosine, manhattan, chebyshev...
save = True             #TRUE TO SAVE FIGURE IN SILHOUETTE/SILHOUETTES

# STATS CONFIG
# USE MODEL_LABEL VARIABLE TO SELECT MODEL TO EXTRACT STATS
model_stats = True     # STATS OF MODEL CLASSIFICATIONS AND DISCARDS
label_stats = False      # STATS OF LABELS OF A CORPUS


# VISUALIZATION CONFIG
model = 'ST2_MPDI_MINTOPICSIZE_30'             # MODELO A VISUALIZAR
formato = 'embeddings'      #REPRESENTAMOS EMBEDDINGS O PROBABILITIES
negros = False               # TRUE PARA ENSEÑAR LOS TOPICS A -1 Y FALSE PARA OCULTARLOS
modo = ''              # topic muestra los nombres de los temas y texto los valores mas significativos, vacio para no mostrar nada
dimensions = 2              # 2 O 3 DIMENSIONES, RECOMENDADO 2
represent = ['pca', 'intertopic']        # INTERTOPIC, PCA UMAP O TSNE, PODEMOS METER VARIOS EN UNA LISTA
n_neighbors = 250           #SOLO PARA UMAP
mode = 'visualize'          # TRAIN PARA GENERAR MODELO, VISUALIZE PARA ENSEÑARLO, SOLO UMAP


# LABELING CONFIG
# MONOLABEL
mono_label_number = 4            # 1 ONLY FIRST LABELING, 2 LABEL +  KNEIGHBORS, 3 LABEL + CENTROIDS, 4 LABEL + KNEIGBORS AND LABEL + CENTROIDS
model_label = 'ST2_MPDI_MINTOPICSIZE_30'                    # MODEL TO RELABEL
relabel = 'random_dict_ST2_MPDI_MINTOPICSIZE_30.txt' # RELABEL DICTIONARY, CHEK LABELS/LABEL_DICT ONCE GENERATED
##### FIRST EVALUATION
label_importance_in_cluster = False
global optimus_dictionary
global effective_dictionary
optimus_dictionary = False
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
multi_label_number = 4  # 1 ONLY FIRST LABELING, 2 MULTI-LABEL +  KNEIGHBORS, 3 LABEL + CENTROIDS, 4 MULTI-LABEL + KNEIGBORS AND LABEL + CENTROIDS
percent = 25    # SECOND LABEL IMPORTANCE PERCENT
kneighbors_labeling = True
kneighbors_eval = True
centroids_labeling = True
centroids_eval = True

## CENTROID DIFFERENCE EVALUATION USE VARIABLE MODEL_LABEL TO SELECT YOUR MODEL TO EVALAUTE
plot_centroid_distances =False         # PLOTS PREDICTIONS AND REAL CENTROIDS PER LABEL
calculate_centroid_distances = False    # DISPLAYS DISTANCES BETWEEN TRUE AND PRED CENTROIDS


# PARA LOS NUEVOS DICCIONARIOS
n_samples = 10
probabilistic_mode = True
random_mode = True
multilabel_dict = 'random_multilabel_ST2_MPDI_MINTOPICSIZE_30_25'  # PROBABILITIES O RANDOM + MODELO + PERCENT

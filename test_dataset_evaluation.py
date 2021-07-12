import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import pandas as pd
import numpy as np
from config import corpus, model_label as model, percent, n_samples, multilabel_dict, relabel, sentence_transformer
from BERTclassifier import getTopics, getTrueLabels, loadPreprocessedText, getEmbeddings
import csv
from collections import Counter
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os.path
from datetime import datetime
from subprocess import call
from labels.escenario_multilabel.performance import bad_pred_multilabel
mode = multilabel_dict.split('_')[0]

## STRING CON DATASET DE TEST
test_corpus = 'TEST_' + '_'.join(corpus.split('_')[1:corpus.__len__()])
label_set = list(dict.fromkeys(getTrueLabels(corpus)))

# CARGAMOS EL TEXTO Y DEFINIMOS EL SENTENCE TRANSFORMER Y CODIFICAMOS EMBEDDINGS
transformer = SentenceTransformer(sentence_transformer)
texto = loadPreprocessedText(test_corpus)
embeddings = transformer.encode(texto)


# DEFINIMOS DATASET DE ENTRENAMIENTO CON ETIQUETAS REALES Y LAS QUE HEMOS PREDICHO Y LOS EMBEDDINGS
df_train = pd.DataFrame(getTrueLabels(corpus), columns=['true'])
df_train['cluster'] = pd.read_csv('./results/%s/labels/centroids_cluster_assignment.txt'%model, header=None)
df_train = df_train.assign(code=[*getEmbeddings(model, 'numpy')])


# HACEMOS LO MISMO PARA EL DE TEST, DEJANDO VACIAS LAS PREDICTS
df_test = pd.DataFrame(getTrueLabels(test_corpus), columns=['true'])
df_test = df_test.assign(code=[*embeddings])
monolabel_pred = []
multilabel_pred = []
clusters_assigned = []


# DEFINIMOS DICCIONARIO CLUSTER -> LABEL
dictionary = pd.read_csv('labels/escenario_multilabel/multilabel_dicts/%s.csv'%multilabel_dict, header=None, delimiter=',', names = ['cluster', 'multilabel'])
dictionary['monolabel'] = pd.read_csv('labels/escenario_monolabel/label_dict/%s'%relabel, header=None, delimiter=',', names = ['cluster', 'monolabel']).monolabel

if not os.path.exists('./results/%s/labels/test_cluster_assignment.txt'%model):
    if not os.path.exists('results/%s/labels/test'%model):
        call('mkdir results/%s/labels/test'%model, shell=True)
    i = 0
    for item in df_test.iterrows():
        # CALCULAMOS DISTANCIA DE UNA MUESTRA DE TEST A TODAS LAS DE TRAIN Y LAS METEMOS EN UNA NUEVA COLUMNA DE DF_TRAIN
        distance = cosine_similarity([item[1]['code']], df_train['code'].tolist())
        df_train['distance'] = distance[0]

        # NOS QUEDAMOS CON LAS 5 MUESTRAS CON MAYOR COSINE_SIMILARITY
        largest = df_train.nlargest(5, 'distance')

        # ASIGNAMOS EL CLUSTER CON MAS PRESENCIA Y LA ETIQUETA ES LA DE ESE CLUSTER (BUSCAMOS EN DICTIONARY)
        cluster_assigned = max(Counter(largest.cluster), key = Counter(largest.cluster).get)
        monolabel = dictionary[dictionary.cluster == cluster_assigned].monolabel
        multilabel = dictionary[dictionary.cluster == cluster_assigned].multilabel
        monolabel_pred.append(monolabel)
        multilabel_pred.append(multilabel)
        clusters_assigned.append(cluster_assigned)

        # CONTADOR DE ITERACIONES
        i += 1
        if i%5000 == 0:
            print(datetime.now().strftime('%H:%M:%S'), i, flush = True)

    df_train = df_train.drop('distance', axis=1)
    # UNA VEZ TERMINADO ASIGNAMOS PREDICCIONES AL DATAFRAME Y LAS GUARDAMOS EN FICHEROS
    df_test['monolabel_pred'] = monolabel_pred
    df_test['multilabel_pred'] = multilabel_pred
    df_test['cluster'] = clusters_assigned
    df_test['cluster'].to_csv('./results/%s/labels/test_cluster_assignment.txt'%model, header=None, index=None)
    df_test['monolabel_pred'].to_csv('./results/%s/labels/test/%s_monolabel_nsamples%s_percent%s.txt'%(model, mode, n_samples, percent), header=None, index=None)
    df_test['multilabel_pred'].to_csv('./results/%s/labels/test/%s_multilabel_nsamples%s_percent%s.txt'%(model, mode, n_samples, percent), header=None, index=None)

else:
    i = 0
    df_test['cluster'] = pd.read_csv('./results/%s/labels/test_cluster_assignment.txt'%model, header=None)
    for item in df_test.iterrows():

        # ASIGNAMOS LABELS DEL CLUSTER CON MAS PRESENCIA
        monolabel_pred.append(dictionary[dictionary.cluster == item[1].cluster]['monolabel'].item())
        multilabel_pred.append(dictionary[dictionary.cluster == item[1].cluster]['multilabel'].item())

        # CONTADOR DE ITERACIONES
        i += 1
        if i%5000 == 0:
            print(datetime.now().strftime('%H:%M:%S'), i, flush = True)

    df_test['monolabel_pred'] = monolabel_pred
    df_test['multilabel_pred'] = multilabel_pred
    df_test['monolabel_pred'].to_csv('./results/%s/labels/test/%s_monolabel_nsamples%s_percent%s.txt'%(model, mode, n_samples, percent), header=None, index=None)
    df_test['multilabel_pred'].to_csv('./results/%s/labels/test/%s_multilabel_nsamples%s_percent%s.txt'%(model, mode, n_samples, percent), header=None, index=None)


print(df_test)

# EVALUACION MONOLABEL

# DEFINIMOS LAS PREDICCIONES Y LAS ETIQUETAS REALES
y_pred = df_test.monolabel_pred
y_true = df_test.true

# PARAMETROS DE EVALUACION
acc = accuracy_score(y_true=y_true, y_pred=y_pred)
prec = precision_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
recall = recall_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
print('MONOLABEL TEST EVALUATION')
print('ACC: %s \nPREC: %s \nRECALL: %s \nF1: %s'%(acc, prec, recall, f1))

# MATRIZ DE CONFUSION
cm = confusion_matrix(y_true, y_pred, label_set)
confusion_m = []
for line in cm:
    confusion_m.append(line)
matrix = pd.DataFrame(confusion_m, index=label_set,columns=label_set)
print(matrix.to_string())

excel = pd.DataFrame([acc,prec,recall,f1], index=['accuracy', 'precision', 'recall', 'f1'], columns = ['monolabel'])

# EVALUACION MULTILABEL
# DEFINIMOS TRUE Y PRED, IGUAL QUE ANTES

true = df_test.true
pred = df_test.multilabel_pred
label_set = list(dict.fromkeys(df_train.true))

# TOTAL DE LABELS REALES
tot = Counter(true)
# SI ALGUNA ETIQUETA NO APARECE EN LAS PREDICTS LA METEMOS CON 0
for label in label_set:
    if label not in tot.keys():
        tot.update({label : 0})
# DATAFRAME DE EVALUACION
evaluation = pd.DataFrame(index=label_set, columns=['precision', 'recall', 'f1'])
evaluation['Total'] = tot.values()
total = len(true)
cluster = df_test.cluster

# HACEMOS CALCULO DE TRUE Y FALSE POSITIVES Y NEGATIVES, APLICANDO PENALIZACIONES SI ASIGNAMOS UN CLUSTER MAL PUESTO
for label in label_set:
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for i in range(len(true)):
        if label == true[i] and true[i] in pred[i]:
            true_positives += 1
            if cluster[i] in bad_pred_multilabel:
               total+=1
        if label == true[i] and true[i] not in pred[i]:
            false_negatives += 1
        if label in pred[i] and true[i] not in pred[i]:
            false_positives += 1
        if label != true[i] and true[i] not in pred[i]:
            true_negatives += 1
            if cluster[i] in bad_pred_multilabel:
                total+=1
    # CALCULAMOS PRECISION RECALL Y F1 PARA CADA ITERACION
    topic_precision = 0
    topic_recall = 0
    topic_f1 = 0
    if true_positives + false_positives != 0:
        topic_precision = true_positives / (true_positives + false_positives)
    if true_positives + false_negatives != 0:
        topic_recall = true_positives / (true_positives + false_negatives)
    if topic_precision + topic_recall != 0:
        topic_f1 = 2 * (topic_precision * topic_recall) / (topic_precision + topic_recall)
    evaluation.loc[label] = [topic_precision, topic_recall, topic_f1, tot.get(label)]
prec, recall, f1 = (0,0,0)
# PRECISION RECALL Y F1 GENERAL
for eval in evaluation.iterrows():
    prec += (eval[1]['precision'] * eval[1]['Total'])/total
    recall += (eval[1]['recall'] * eval[1]['Total'])/total
    f1 += (eval[1]['f1'] * eval[1]['Total'])/total
i = 0
# ACCURACY
for j in range(len(true)):
    if true[j] in pred[j]:
        i += 1
acc = i / total
print('Acc: %s \t Prec: %s \t Recall: %s \t F1: %s'%(acc,prec,recall,f1))

if not os.path.exists('./results/%s/test'%model):
     call("mkdir ./results/%s/test"%model, shell=True)


print(excel)
excel['multilabel'] = list([acc,prec,recall,f1])
excel.to_excel('./results/%s/test/%s_nsamples%s_percent%s.xlsx'%(model, mode, n_samples, percent))
print(excel)

import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
from sentence_transformers import SentenceTransformer
from config import sentence_transformer, relabel, corpus, model_label as model, optimus_dictionary, effective_dictionary
from BERTclassifier import getTopics, loadPreprocessedText
import pandas as pd
import numpy as np
from collections import Counter
from subprocess import call
global power_on
power_on = True

call('mkdir /results/%s/labels'%model, shell=True)
call('mkdir results/%s/labels/monolabel/'%model, shell=True)
call('mkdir results/%s/labels/multilabel/'%model, shell=True)

def getTopicList(corpus):
    # GETS ACCURATE LABELS FOR EVERY INPUT
    topics = []
    for line in open('./corpus/preprocessed/preprocess_%s'%corpus, 'r', encoding='utf-8'):
        topics.append(line.split('\t')[1].strip())
    return topics
solutions = getTopicList(corpus)

def effectiveDictionary(corpus, model):
    # BUILDS THE MOST EFFECTIVE DICTIONARY DEPENDING ON THE TOPIC WITH MOST PRESENCE IN EACH CLUSTER
    topics = getTopicList(corpus)
    clusters = getTopics(model)
    n_clusters = list(dict.fromkeys(clusters))
    f = open('./labels/escenario_monolabel/label_dict/effective_dictionary_%s.txt'%model, 'w', encoding='utf-8')
    for n_cluster in n_clusters:
        if n_cluster == -1:
            f.write('-1,descarte\n')
            continue
        lista_topics = []
        for i in range(len(clusters)):
            if n_cluster == clusters[i]:
                lista_topics.append(topics[i])
        f.write('%s,%s\n'%(n_cluster,Counter(lista_topics).most_common()[0][0]))
    f.close()
    print('DONE EFFECTIVE DICTIONARY')

def optimusDictionary(corpus, model):
    # BUILDS THE MOST OPTIMUS DICTIONARY DEPENDING ON THE TOPIC WITH MOST PRESENCE IN EACH CLUSTER
    topics = getTopicList(corpus)
    clusters = getTopics(model)
    n_clusters = list(dict.fromkeys(clusters))
    lista_perc = [1, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.20, 0.15, 0.10, 0.05, 0]
    optimus_dict = {}
    w_prec = 0
    percent = 0
    for perc in lista_perc:
        new_dict = {}
        new_w_prec = 0
        for n_cluster in n_clusters:
            if n_cluster == -1:
                new_dict.update({n_cluster : 'descarte'})
                continue
            lista_topics = []
            for i in range(len(clusters)):
                if n_cluster == clusters[i]:
                    lista_topics.append(topics[i])
            primero = Counter(lista_topics).most_common()[0][1]
            if Counter(lista_topics).most_common().__len__() > 1:
                segundo = Counter(lista_topics).most_common()[1][1]
            else:
                segundo = 0
            if Counter(lista_topics).most_common()[0][0] in ['política', 'otros'] and segundo != 0 and Counter(lista_topics).most_common()[1][0] not in ['política', 'otros']:
                if (primero-segundo)/len(lista_topics) < perc:
                    new_dict.update({n_cluster : Counter(lista_topics).most_common()[1][0]})
                else:
                    new_dict.update({n_cluster : Counter(lista_topics).most_common()[0][0]})
            else:
                new_dict.update({n_cluster : Counter(lista_topics).most_common()[0][0]})
        new_w_prec = w_acc(new_dict, corpus, model)
        #print(new_w_prec)
        if w_prec <= new_w_prec:
            optimus_dict = new_dict
            w_prec = new_w_prec
            percent = perc
    print('Final perc = %s \nFinal w_prec = %s'%(percent, w_prec))
    f = open('./labels/escenario_monolabel/label_dict/optimus_dictionary_%s.txt'%model, 'w', encoding='utf-8')
    for i in range(optimus_dict.values().__len__()):
        f.write(str(list(optimus_dict.keys())[i]) + ',' + list(optimus_dict.values())[i] + '\n')
    f.close()
    print('DONE OPTIMUS DICTIONARY')

def w_acc(dictionary, corpus, model):
    y_pred = getTopics(model)
    for i in range(len(y_pred)):
        y_pred[i] = dictionary.get(y_pred[i])
    y_true = getTopicList(corpus)
    topic_set = list(dict.fromkeys(y_true))
    accuracy_list = []
    for topic in topic_set:
        corrects = 0
        for i in range(len(y_pred)):
            if topic in y_pred[i] and y_pred[i] in y_true[i]:
                corrects += 1
        accuracy_list.append(corrects / Counter(y_true)[topic])
    return sum(accuracy_list)/len(accuracy_list)

if optimus_dictionary:
    optimusDictionary(corpus, model)
optimus_dictionary = False

if effective_dictionary:
    effectiveDictionary(corpus, model)
effective_dictionary = False

if power_on:
    topics = getTopicList(corpus)
    text = loadPreprocessedText(corpus)

def relabelDict(relabel):
    # GETS THE DICTIONARY OF RELABEL
    df = pd.read_csv('./labels/escenario_monolabel/label_dict/%s'%relabel, header=None, delimiter=',')
    clusters = list(df[0])
    topics = list(df[1])
    print('GOT DICTIONARY')
    return dict(zip(clusters, topics))


def reclassify(relabel, model):
    # PUTS LABELS TO EVERY CLUSTER DEPENDING ON THE DICTIONARY
    diccionario = relabelDict(relabel)
    nuevo = getTopics(model=model)
    new_topic_list = []
    for line in nuevo:
        new_topic_list.append(diccionario.get(line))
    print('PUT LABELS IN CLUSTERS')
    return new_topic_list


if power_on:
    sbert_model = SentenceTransformer(sentence_transformer)
    predicts = reclassify(relabel, model)

def get_label_set(corpus):
    return list(dict.fromkeys(getTopicList(corpus)))


def get_prediction_indexes():
    index_list = []
    for i in range(len(solutions)):
        if 'descart' not in predicts[i]:
            index_list.append(i)
    return index_list


if power_on:
    index_predict = get_prediction_indexes()

def get_topic_prediction_list(topic):
    topic_list = []
    indexes = index_predict
    for index in indexes:
        if topic in predicts[index]:
            topic_list.append(text[index])
    return topic_list

def get_prediction_embeddings_codification(corpus, label_set):
    prediction_embeddings_codification = {}
    for label in label_set:
        encoding = sbert_model.encode(get_topic_prediction_list(label))
        prediction_embeddings_codification.update({label : encoding})
    return prediction_embeddings_codification

power_on = False

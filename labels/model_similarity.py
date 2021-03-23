import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import re
from sklearn.metrics.pairwise import cosine_similarity
from labels.labels_evaluation_v1 import corpus, model, relabel, getTopicList, reclassify, model_similarity, label_in_model_similarity
from sentence_transformers import SentenceTransformer
from BERTclassifier import getSamples, getTopics, loadPreprocessedText
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix
from datetime import datetime
from xlrd import open_workbook
from xlutils.copy import copy
import xlwt

#sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
#sbert_model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
sbert_model = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
predicts = reclassify(relabel, model)


def get_accurate_indexes():
    index_list = []
    solutions = getTopicList(corpus)
    for i in range(len(solutions)):
        if predicts[i] in solutions[i]:
            index_list.append(i)
    return index_list

index_correct = get_accurate_indexes()
text = loadPreprocessedText(corpus)
topics = getTopicList(corpus)

def get_topic_accurate_list(topic):
    topic_list = []
    indexes = index_correct
    for index in indexes:
        if topic in topics[index]:
            topic_list.append(text[index])
    return topic_list


def determine_proximity_to_topic(input, topic):
    topic_accurate_list = get_topic_accurate_list(topic)
    input = [sbert_model.encode(input)]
    topics = sbert_model.encode(topic_accurate_list)
    if len(topics) == 0:
        return 0
    similarity =  cosine_similarity(input, topics)
    if type(np.max(similarity)) == np.float32:
        return np.max(similarity)
    else:
        return np.max(similarity)[0]


label_set = ['SDG', 'cine', 'deportes', 'economía', 'entretenimiento', 'fútbol', 'hoteles', 'literatura', 'marcas',
             'música', 'otros', 'política', 'restaurantes', 'tecnología']

accurate_embeddings_codification = {}
for label in label_set:
    encoding = sbert_model.encode(get_topic_accurate_list(label))
    accurate_embeddings_codification.update({label : encoding})




def determine_proximity_to_topic(input, topic):
    input = [sbert_model.encode(input)]
    topics = accurate_embeddings_codification.get(topic)
    if len(topics) == 0:
        return -1
    similarity =  cosine_similarity(input, topics)
    if type(np.max(similarity)) == np.float32:
        return np.max(similarity)
    else:
        return np.max(similarity)[0]



def label_discard(discard):
    proximities = []
    for label in label_set:
        proximities.append(determine_proximity_to_topic(discard, label))
    #print(label_set[np.argmax(proximities)])
    return label_set[np.argmax(proximities)]


def get_discard_indexes():
    indexes = []
    for i in range(len(predicts)):
        if 'desc' in predicts[i]:
            indexes.append(i)
    return indexes



def label():
    for i in range(len(predicts)):
        if 'desc' in predicts[i]:
            predicts[i] = label_discard(text[i])
            if i%7500 == 0:
                print(datetime.now().strftime('%H:%M:%S'), flush=True)
                print(i, flush=True)
    f = open('./results/%s/predicts.txt'%model, 'w', encoding='utf-8')
    for line in predicts:
        f.write(line.replace('\n', "") + '\n')
    f.close()

def get_final_predicts():
    final_predicts = []
    for line in open('./results/%s/predicts.txt'%model, 'r', encoding='utf-8'):
        final_predicts.append(line.replace('\n', ""))
    return final_predicts

def get_discard_labels():
    clusters = getTopics(model)
    guess = get_final_predicts()
    discard_labels = []
    for cluster in clusters:
        if cluster == -1:
            discard_labels.append(guess[clusters.index(cluster)])

def discards_evaluation():
    y_pred = []
    y_true = []
    clusters = getTopics(model)
    final_predicts = get_final_predicts()
    print(final_predicts[0:10])
    print(topics[0:10])
    for i in range(len(clusters)):
        if clusters[i] == -1:
            y_pred.append(final_predicts[i])
            y_true.append(topics[i])
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prec = precision_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    recall = recall_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    print('ACC: %s \nPREC: %s \nRECALL: %s \nF1: %s'%(acc, prec, recall, f1), flush=True)
    confusion_m = confusion_matrix(y_true, y_pred, label_set)
    print(pd.DataFrame(confusion_m, index=label_set,columns=label_set).to_string(), flush=True)
    matrix = pd.DataFrame(confusion_m, index=label_set,columns=label_set)
    matrix.to_excel('./results/%s/evaluation/kneighbors_evaluation.xls'%model, columns=label_set, index=label_set, startcol=1, startrow=1, merge_cells=True)
    rb = open_workbook('./results/%s/evaluation/kneighbors_evaluation.xls'%model)
    wb = copy(rb)
    w_sheet = wb.get_sheet(0)
    w_sheet.write(0, 0, model)
    w_sheet.write(len(label_set) + 2, 1, 'ACC', xlwt.easyxf('font: bold 1'))
    w_sheet.write(len(label_set) + 3, 1, 'PREC', xlwt.easyxf('font: bold 1'))
    w_sheet.write(len(label_set) + 4, 1, 'RECALL', xlwt.easyxf('font: bold 1'))
    w_sheet.write(len(label_set) + 5, 1, 'F1', xlwt.easyxf('font: bold 1'))
    w_sheet.write(len(label_set) + 2, 2, acc)
    w_sheet.write(len(label_set) + 3, 2, prec)
    w_sheet.write(len(label_set) + 4, 2, recall)
    w_sheet.write(len(label_set) + 5, 2, f1)
    wb.save('./results/%s/evaluation/kneighbors_evaluation.xls'%model)


def complete_evaluation():
    y_true = topics
    y_pred = get_final_predicts()
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prec = precision_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    recall = recall_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=label_set, average='weighted')
    print('ACC: %s \nPREC: %s \nRECALL: %s \nF1: %s'%(acc, prec, recall, f1), flush=True)
    confusion_m = confusion_matrix(y_true, y_pred, label_set)
    print(pd.DataFrame(confusion_m, index=label_set,columns=label_set).to_string(), flush=True)
    matrix = pd.DataFrame(confusion_m, index=label_set,columns=label_set)
    matrix.to_excel('./results/%s/evaluation/kneighbors_final_evaluation.xls'%model, columns=label_set, index=label_set, startcol=1, startrow=1, merge_cells=True)
    rb = open_workbook('./results/%s/evaluation/kneighbors_final_evaluation.xls'%model)
    wb = copy(rb)
    w_sheet = wb.get_sheet(0)
    w_sheet.write(0, 0, model)
    w_sheet.write(len(label_set) + 2, 1, 'ACC', xlwt.easyxf('font: bold 1'))
    w_sheet.write(len(label_set) + 3, 1, 'PREC', xlwt.easyxf('font: bold 1'))
    w_sheet.write(len(label_set) + 4, 1, 'RECALL', xlwt.easyxf('font: bold 1'))
    w_sheet.write(len(label_set) + 5, 1, 'F1', xlwt.easyxf('font: bold 1'))
    w_sheet.write(len(label_set) + 2, 2, acc)
    w_sheet.write(len(label_set) + 3, 2, prec)
    w_sheet.write(len(label_set) + 4, 2, recall)
    w_sheet.write(len(label_set) + 5, 2, f1)
    wb.save('./results/%s/evaluation/kneighbors_final_evaluation.xls'%model)

if model_similarity:
    print('hola', flush=True)
    if label_in_model_similarity:
        print('label', flush=True)
        label()
    discards_evaluation()
    complete_evaluation()

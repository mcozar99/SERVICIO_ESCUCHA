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
from labels.labels_evaluation_v1 import corpus, model, relabel, getTopicList, evaluation, reclassify
from sentence_transformers import SentenceTransformer
from BERTclassifier import getEmbedding


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


def get_topic_accurate_list(topic):
    topic_list = []
    text = getEmbedding(corpus)
    topics = getTopicList(corpus)
    indexes = get_accurate_indexes()
    for index in indexes:
        if topic in topics[index]:
            print(index)
            topic_list.append(text[index])
    return topic_list

def determine_proximity_to_topic(input, topic):
    input = sbert_model.encode(input)
    topics = sbert_model.encode(get_topic_accurate_list(topic))
    similarity =  cosine_similarity(input, topics)
    return np.max(similarity)
#pairwise_similarities=cosine_similarity(document_embeddings)


print(determine_proximity_to_topic('LAS NUEVAS ZAPATILLAS DE NIKE SON UNA LOCURA', 'marcas'))
print(determine_proximity_to_topic('LAS NUEVAS ZAPATILLAS DE NIKE SON UNA LOCURA', 'política'))
print(determine_proximity_to_topic('LAS NUEVAS ZAPATILLAS DE NIKE SON UNA LOCURA', 'deportes'))


""" cosine_similarity([sbert_model.encode('Las zapatillas rebajadas son una locura')], document_embeddings)"""

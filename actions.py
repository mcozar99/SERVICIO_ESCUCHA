import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import silhouette.silhouette as silhouette
import BERTclassifier as bc
from bertopic import BERTopic as bt


action = 'train'         # SILHOUETTE, INFO OR TRAIN
#################################################
model_name = 'EMBEDDING_CHILE'
min_topic_sizes = [100]
################################################# GENERAL
corpus = 'CORPUS_SERVICIO_ESCUCHA.txt'
iterations = 10                                #Number of tries to improve %classified
################################################# TRAIN
info = 'topics'          # TOPICS O INFO
n_topics = 10            # Number of topics to show
################################################# INFO
metric = 'cosine'       #metric to measure results: euclidean, cosine, manhattan, chebyshev...
save = True             #TRUE TO SAVE FIGURE IN SILHOUETTE/SILHOUETTES
################################################# SILHOUETTE

print('SELECTED: %s'%action)

if action == 'train':
     for min_topic_size in min_topic_sizes:
          bc.train(corpus=corpus, name=model_name, min_topic_size=min_topic_size, iterations=iterations)

elif action == 'info':
     if info == 'info':
          for min_topic_size in min_topic_sizes:
               bc.getInfo(name='%s_MINTOPICSIZE_%s'%(model_name, min_topic_size))
     elif info == 'topics':
          for min_topic_size in min_topic_sizes:
               bc.getTopTopics(name='%s_MINTOPICSIZE_%s'%(model_name, min_topic_size), n=n_topics)

elif action == 'silhouette':
     for min_topic_size in min_topic_sizes:
          silhouette.silhouette(model='%s_MINTOPICSIZE_%s'%(model_name, min_topic_size), metric=metric, save=save)

bc.visualizeModel('TEST_MINTOPICSIZE_100')
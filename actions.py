import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import silhouette.silhouette as silhouette
import BERTclassifier as bc
from bertopic import BERTopic as bt
from config import actions, model_name, min_topic_sizes, corpus, sentence_transformer, iterations, info, n_topics, metric, save, label_number



print('SELECTED: %s'%actions)

if 'train' in actions:
     for min_topic_size in min_topic_sizes:
          bc.train(corpus=corpus, name=model_name, min_topic_size=min_topic_size, iterations=iterations, sentence_transformer=sentence_transformer)

if 'info' in actions:
     if info == 'info':
          for min_topic_size in min_topic_sizes:
               bc.getInfo(name='%s_MINTOPICSIZE_%s'%(model_name, min_topic_size))
     elif info == 'topics':
          for min_topic_size in min_topic_sizes:
               bc.getTopTopics(name='%s_MINTOPICSIZE_%s'%(model_name, min_topic_size), n=n_topics)

if 'silhouette' in actions:
     for min_topic_size in min_topic_sizes:
          silhouette.silhouette(model='%s_MINTOPICSIZE_%s'%(model_name, min_topic_size), metric=metric, save=save)


if 'complete_evaluation' in actions:
    # 1 ONLY FIRST LABELING, 2 LABEL +  KNEIGHBORS, 3 LABEL + CENTROIDS, 4 LABEL + KNEIGBORS AND LABEL + CENTROIDS
    if label_number == 1:
        import labels.labels_evaluation_v1
    elif label_number == 2:
        import labels.labels_evaluation_v1
        import labels.model_similarity
    elif label_number == 3:
        import labels.labels_evaluation_v1
        import labels.centroids
    elif label_number == 4:
        import labels.labels_evaluation_v1
        import labels.model_similarity
        import labels.centroids

if 'visualization' in actions:
    import visualization.visualize_conf

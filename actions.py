import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import silhouette.silhouette as silhouette
import BERTclassifier as bc
from bertopic import BERTopic as bt
from config import actions, model_name, min_topic_sizes, corpus, sentence_transformer, iterations, info, n_topics, metric, save, mono_label_number, multi_label_number


print('SELECTED: %s'%actions)

if 'train' in actions:
     for min_topic_size in min_topic_sizes:
          bc.train(corpus=corpus, name=model_name, min_topic_size=min_topic_size, iterations=iterations, sentence_transformer=sentence_transformer)

if 'info' in actions:
     if 'info' in info:
          for min_topic_size in min_topic_sizes:
               bc.getInfo(name='%s_MINTOPICSIZE_%s'%(model_name, min_topic_size))
     if 'topics' in info:
          for min_topic_size in min_topic_sizes:
               bc.getTopTopics(name='%s_MINTOPICSIZE_%s'%(model_name, min_topic_size), n=n_topics)

if 'silhouette' in actions:
     for min_topic_size in min_topic_sizes:
          silhouette.silhouette(model='%s_MINTOPICSIZE_%s'%(model_name, min_topic_size), metric=metric, save=save)

if 'stats' in actions:
    import stats

if 'monolabel_evaluation' in actions:
    # 1 ONLY FIRST LABELING, 2 LABEL +  KNEIGHBORS, 3 LABEL + CENTROIDS, 4 LABEL + KNEIGBORS AND LABEL + CENTROIDS
    if mono_label_number == 1:
        import labels.escenario_monolabel.labels_evaluation_v1
    elif mono_label_number == 2:
        import labels.escenario_monolabel.labels_evaluation_v1
        import labels.escenario_monolabel.model_similarity
    elif mono_label_number == 3:
        import labels.escenario_monolabel.labels_evaluation_v1
        import labels.escenario_monolabel.centroids
    elif mono_label_number == 4:
        import labels.escenario_monolabel.labels_evaluation_v1
        import labels.escenario_monolabel.model_similarity
        import labels.escenario_monolabel.centroids

if 'multilabel_evaluation' in actions:
    if multi_label_number == 1:
        import labels.escenario_multilabel.multilabel_functions

if 'visualization' in actions:
    import visualization.visualize_conf

if 'centroid_evaluation' in actions:
    import centroid_differences.labels_centroid_eval

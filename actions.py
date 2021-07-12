import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import silhouette.silhouette as silhouette
import BERTclassifier as bc
from bertopic import BERTopic as bt
from config import actions, model_name, min_topic_sizes, corpus, sentence_transformer, iterations, info, n_topics, metric, save, model_label as model, percent, n_samples
from subprocess import call

print('SELECTED: %s'%actions)
print('PERCENT: %s,\t N_SAMPLES: %s'%(percent, n_samples))
print(model)
print(sentence_transformer)

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

if 'label' in actions:
    import labels.dictionaries
    import labels.escenario_monolabel.first_monolabel_evaluation
    import labels.escenario_multilabel.first_multilabel_evaluation
    import labels.label
    import labels.escenario_monolabel.kneighbors_evaluation
    import labels.escenario_monolabel.centroids_evaluation
    import labels.escenario_multilabel.kneighbors_evaluation
    import labels.escenario_multilabel.centroids_evaluation
    import labels.final_results

if 'visualization' in actions:
    import visualization.visualize_conf

if 'centroid_evaluation' in actions:
    import centroid_differences.labels_centroid_eval
if 'test' in actions:
    import test_dataset_evaluation

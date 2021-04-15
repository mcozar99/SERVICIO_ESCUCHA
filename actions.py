import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import silhouette.silhouette as silhouette
import BERTclassifier as bc
from bertopic import BERTopic as bt
from config import actions, model_name, min_topic_sizes, corpus, sentence_transformer, iterations, info, n_topics, metric, save, mono_label_number, multi_label_number, model_label as model, percent, n_samples
from subprocess import call

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


if 'multilabel_evaluation' in actions:
    import labels.dictionaries
    call('mkdir ./results/%s/labels'%model, shell=True)
    call('mkdir ./results/%s/labels/multilabel/'%model, shell=True)
    if multi_label_number == 1:
        import labels.escenario_multilabel.multilabel_functions
    if multi_label_number == 2:
        import labels.escenario_multilabel.multilabel_functions
        import labels.escenario_multilabel.kneighbors
    if multi_label_number == 3:
        import labels.escenario_multilabel.multilabel_functions
        import labels.escenario_multilabel.centroides
    if multi_label_number == 4:
        import labels.escenario_multilabel.multilabel_functions
        import labels.escenario_multilabel.centroides
        import labels.escenario_multilabel.kneighbors


if 'monolabel_evaluation' in actions:
    import labels.dictionaries
    # 1 ONLY FIRST LABELING, 2 LABEL +  KNEIGHBORS, 3 LABEL + CENTROIDS, 4 LABEL + KNEIGBORS AND LABEL + CENTROIDS
    call('mkdir /results/%s/labels'%model, shell=True)
    call('mkdir results/%s/labels/monolabel/'%model, shell=True)
    if mono_label_number == 1:
        import labels.escenario_monolabel.labels_evaluation_v1
    elif mono_label_number == 2:
        import labels.escenario_monolabel.labels_evaluation_v1
        import labels.escenario_monolabel.model_similarity
    elif mono_label_number == 3:
        import labels.escenario_monolabel.labels_evaluation_v1
        import labels.escenario_monolabel.centroides
    elif mono_label_number == 4:
        import labels.escenario_monolabel.labels_evaluation_v1
        import labels.escenario_monolabel.centroides
        import labels.escenario_monolabel.model_similarity
    import labels.final_results

if 'visualization' in actions:
    import visualization.visualize_conf

if 'centroid_evaluation' in actions:
    import centroid_differences.labels_centroid_eval
print('PERCENT: %s,\t N_SAMPLES: %s'%(percent, n_samples))
print(model)

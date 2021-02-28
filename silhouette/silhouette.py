import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')

from BERTclassifier import getTopics, getProbabilities
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from datetime import datetime

def silhouette(model, metric, save):
    topics = getTopics('%s'%model)
    n_topics = list(dict.fromkeys(topics)).__len__()
    list_topics = list(dict.fromkeys(topics))
    probs = getProbabilities('%s'%model, 'numpy')
    print(datetime.today(), ' - Got params')
    # The silhouette_score gives the average value for all the samples.
    silhouette_avg = silhouette_score(probs, topics, metric=metric)
    print("For these %s topics."%n_topics ,"The average silhouette_score is:", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(probs, topics, metric=metric)
    print(datetime.today(), ' - Calculated Silhouette, Plotting')
    # Create a subplot with 1 row and 2 columns
    plt.figure(figsize=(16, 7))
    y_lower = 10
    for topic in list_topics:
        #if topic == -1:
        #    continue
        ith_cluster_silhouette_values = []
        indexes = []
        for i in range(len(topics)):
            if topic == topics[i]:
                indexes.append(i)
        for index in indexes:
            ith_cluster_silhouette_values.append(sample_silhouette_values[index])
        ith_cluster_silhouette_values = np.array(ith_cluster_silhouette_values)
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        color = cm.nipy_spectral(float(topic) / n_topics)
        y_upper = y_lower + size_cluster_i
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(topic))
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
        plt.title("Silhouette plot for model %s with metric %s."%(model,metric))
        plt.xlabel("Average silhouette = %s"%silhouette_avg)
        plt.ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        plt.axvline(x=silhouette_avg, color="red", linestyle="--")

        plt.yticks([])  # Clear the yaxis labels / ticks

        if save:
            plt.savefig('./silhouette/silhouettes/silhouette_%s_metric_%s.pdf'%(model,metric))
    plt.show()


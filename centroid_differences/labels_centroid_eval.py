import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from BERTclassifier import getLabels, loadPreprocessedText
from actions import corpus, model_name, sentence_transformer
from labels.centroids import k_means

sbert_model = SentenceTransformer(sentence_transformer)
true_labels = getLabels(corpus)
predictions = pd.read_csv('./results/%s/predicts.txt'%model_name).values.tolist()
embeddings = sbert_model.encode(loadPreprocessedText(corpus))

print(embeddings[0])
print(true_labels[0:10])
print(predictions[0:10])

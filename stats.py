import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import numpy as np
import pandas as pd
import csv
from config import corpus, model_label as model, model_stats, label_stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from textstat import fernandez_huerta
from sentiment_analysis_spanish import sentiment_analysis
from subprocess import call
import os.path

sentiment = sentiment_analysis.SentimentAnalysisSpanish()
def sentimiento(text):
    return sentiment.sentiment(text)


df = pd.read_csv('./corpus/preprocessed/preprocess_%s'%corpus, delimiter='\t', header=None, names=['index', 'label', 'text'], quoting=csv.QUOTE_NONE, error_bad_lines=False)

if label_stats:
    if not os.path.exists('./corpus/stats'):
        call('mkdir ./corpus/stats', shell=True)
    call('mkdir ./corpus/stats/%s'%corpus.replace('.txt', ''), shell=True)

    label_set = list(dict.fromkeys(df['label'].str.strip()))

    for label in label_set:
        label_df = df[df.label.str.contains(label,case=True)]
        fig, axes = plt.subplots(4, 1, sharex=False, figsize=(20, 20))
        fig.suptitle('Stats for label %s'%label)

        # ROW 1
        sns.histplot(data=label_df['text'].str.len(), ax = axes[0], stat='count', palette = sns.color_palette('hls'))
        axes[0].axvline(x=label_df['text'].str.len().mean(), color="red", linestyle="--", label='mean')
        axes[0].set_xlim([0, 300])
        axes[0].set_title('Number of characters in samples')
        axes[0].set_xlabel('MEAN: %s'%label_df['text'].str.len().mean())

        # ROW 2
        sns.histplot(data=label_df['text'].str.split().apply(len), ax = axes[1], stat='count', palette = sns.color_palette('hls'))
        axes[1].axvline(x=label_df['text'].str.split().apply(len).mean(), color="red", linestyle="--", label='mean')
        axes[1].set_title('Number of words in samples')
        axes[1].set_xlim([0, 50])
        axes[1].set_xlabel('MEAN: %s'%label_df['text'].str.split().apply(len).mean())

        # ROW 3
        sns.histplot(data=label_df['text'].apply(lambda x : fernandez_huerta(x)), ax = axes[2], stat='count')
        axes[2].axvline(x=label_df['text'].apply(lambda x : fernandez_huerta(x)).mean(), color='red', linestyle='--', label='mean')
        axes[2].set_title('Readability of samples')
        axes[2].set_xlim([-130, 130])
        axes[2].set_xlabel('MEAN: %s'%label_df['text'].apply(lambda x : fernandez_huerta(x)).mean())

        # ROW 4
        sns.histplot(data=label_df['text'].apply(lambda x : sentimiento(x)), ax = axes[3], stat = 'count')
        axes[3].axvline(x = label_df['text'].apply(lambda x : sentimiento(x)).mean(),  color="red", linestyle="--", label='mean')
        axes[3].set_title('Sentiment of samples')
        axes[3].set_xlim([0, 1])
        axes[3].set_xlabel('MEAN: %s'%label_df['text'].apply(lambda x : sentimiento(x)).mean())

        plt.tight_layout()

        plt.savefig('./corpus/stats/%s/%s.png'%(corpus.replace('.txt', ''), label))


if model_stats:
    df['cluster'] = pd.read_csv('./results/%s/topics.txt'%model, header=None, names=['cluster'], quoting=csv.QUOTE_NONE, error_bad_lines=False)

    discards = df[df['cluster'] == -1]
    classified = df[df['cluster'] != -1]

    fig, axes = plt.subplots(4, 2, sharex=False, figsize=(28, 20))
    fig.suptitle('Stats for model %s'%model)

    # FIRST ROW: NUM CHARACTERS
    sns.histplot(data=classified['text'].str.len(), ax = axes[0,0], stat='count', palette = sns.color_palette('hls'))
    axes[0, 0].axvline(x=classified['text'].str.len().mean(), color="red", linestyle="--", label='mean')
    axes[0, 0].set_title('Number of char classified')
    axes[0, 0].set_xlim([0, 300])
    axes[0, 0].set_xlabel('MEAN: %s'%classified['text'].str.len().mean())

    sns.histplot(data=discards['text'].str.len(), ax = axes[0,1], stat='count', palette = sns.color_palette('hls'))
    axes[0, 1].axvline(x=discards['text'].str.len().mean(), color="red", linestyle="--", label='mean')
    axes[0, 1].set_title('Number of char discards')
    axes[0, 1].set_xlim([0, 300])
    axes[0, 1].set_xlabel('MEAN: %s'%discards['text'].str.len().mean())

    # SECOND ROW: NUM WORDS
    sns.histplot(data=classified['text'].str.split().apply(len), ax = axes[1, 0], stat='count', palette = sns.color_palette('hls'))
    axes[1, 0].axvline(x=classified['text'].str.split().apply(len).mean(), color="red", linestyle="--", label='mean')
    axes[1, 0].set_title('Number of words classified')
    axes[1, 0].set_xlim([0, 50])
    axes[1, 0].set_xlabel('MEAN: %s'%classified['text'].str.split().apply(len).mean())

    sns.histplot(data=discards['text'].str.split().apply(len), ax = axes[1, 1], stat='count', palette = sns.color_palette('hls'))
    axes[1, 1].axvline(x=discards['text'].str.split().apply(len).mean(), color="red", linestyle="--", label='mean')
    axes[1, 1].set_title('Number of words discards')
    axes[1, 1].set_xlim([0, 50])
    axes[1, 1].set_xlabel('MEAN: %s'%discards['text'].str.split().apply(len).mean())


    # THIRD ROW: READABILITY
    sns.histplot(data=classified['text'].apply(lambda x : fernandez_huerta(x)), ax = axes[2,0], stat='count')
    axes[2, 0].axvline(x=classified['text'].apply(lambda x : fernandez_huerta(x)).mean(), color='red', linestyle='--', label='mean')
    axes[2, 0].set_title('Readability of classified')
    axes[2, 0].set_xlim([-130, 130])
    axes[2, 0].set_xlabel('MEAN: %s'%classified['text'].apply(lambda x : fernandez_huerta(x)).mean())

    sns.histplot(data=discards['text'].apply(lambda x : fernandez_huerta(x)), ax = axes[2,1], stat='count')
    axes[2, 1].axvline(x=discards['text'].apply(lambda x : fernandez_huerta(x)).mean(), color='red', linestyle='--', label='mean')
    axes[2, 1].set_title('Readability of discards')
    axes[2, 1].set_xlim([-130, 130])
    axes[2, 1].set_xlabel('MEAN: %s'%discards['text'].apply(lambda x : fernandez_huerta(x)).mean())

    # FOURTH ROW: POLARITY
    sns.histplot(data=classified['text'].apply(lambda x : sentimiento(x)), ax = axes[3, 0], stat = 'count')
    axes[3, 0].axvline(x = classified['text'].apply(lambda x : sentimiento(x)).mean(),  color="red", linestyle="--", label='mean')
    axes[3, 0].set_title('Sentiment of classified')
    axes[3, 0].set_xlim([0, 1])
    axes[3, 0].set_xlabel('MEAN: %s'%classified['text'].apply(lambda x : sentimiento(x)).mean())

    sns.histplot(data=discards['text'].apply(lambda x : sentimiento(x)), ax = axes[3, 1], stat = 'count')
    axes[3, 1].axvline(x = discards['text'].apply(lambda x : sentimiento(x)).mean(),  color="red", linestyle="--", label='mean')
    axes[3, 1].set_title('Sentiment of discards')
    axes[3, 1].set_xlim([0, 1])
    axes[3, 1].set_xlabel('MEAN: %s'%discards['text'].apply(lambda x : sentimiento(x)).mean())

    plt.tight_layout()

    plt.savefig('./results/%s/stats.png'%model)


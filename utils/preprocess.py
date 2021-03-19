import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import pandas as pd
import numpy as np
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.dicts.emoticons import emoticons
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from BERTclassifier import getSamples
from labels.labels_evaluation_v1 import getTopicList


corpus = 'CORPUS_SERVICIO_ESCUCHA.txt'

samples = getSamples(corpus)
classes = getTopicList(corpus)
preprocessing = True
stopwords_removal = True

def text_preprocessing():
    preprocessor = TextPreProcessor(
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date', 'number'],
        annotate={"hashtag", "elongated", "allcaps", "repeated", 'emphasis', 'censored'},
        all_caps_tag="wrap",
        fix_text=True,
        segmenter="twitter_2018",
        corrector="twitter_2018",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True, verbose=True).tokenize,
        #tokenizer=None,
        dicts = [emoticons]
    ).pre_process_doc

    def preprocess(name, dataset):
        desc = "PreProcessing dataset {}...".format(name)

        data = [preprocessor(x)
                for x in tqdm(dataset, desc=desc)]
        return data

    return preprocess

class TextPreprocessing:
    def __init__(self,
                 samples,
                 classes,
                 max_length=0,
                 name=None,
                 preprocess=None):
        """

        Args:
            X (): List of training samples
            y (): List of training labels
            name (str): the name of the dataset. It is needed for caching.
                if None then caching is disabled.
            max_length (int): the max length for each sentence.
                if 0 then use the maximum length in the dataset
        """
        self.data = samples
        self.labels = classes
        self.name = name

        if preprocess is not None:
            self.preprocess = preprocess

        self.set_max_length(max_length)

    def set_max_length(self, max_length):
        # if max_length == 0, then set max_length
        # to the maximum sentence length in the dataset
        if max_length == 0:
            self.max_length = max([len(x) for x in self.data])
        else:
            self.max_length = max_length


def remove_stopwords(sentence_tokens, language):
    #return [ token for token in nltk.word_tokenize(sentence) if token.lower() not in stopwords.words(language) ]
    return [ token for token in sentence_tokens if token.lower() not in stopwords.words(language) ]

if preprocessing:
    preprocessor = text_preprocessing() #twitter_preprocess()
    preprocessed_samples = TextPreprocessing(samples,
                 classes,
                 #max_length=0,
                 name="BERTopic",
                 preprocess=preprocessor)

    samples = preprocessed_samples.data
    samples_detokenized = []
    num_stopwords_removed = 0
    num_sentences_with_stopwords = 0
    num_words = 0
    for item in samples:
        aux_sample = item
        # STOPWORDS REMOVAL
        if stopwords_removal:
            aux_num_words_prev = len(aux_sample)
            num_words = num_words + aux_num_words_prev
            aux_sample = aux_sample.replace('\n', " ").replace('\t', " ").split(' ') #CON ESTO SI QUE VA
            aux_sample = remove_stopwords(aux_sample, 'spanish') #### ESTO NO VA
            aux_num_words_post = len(aux_sample)
            aux_num_stopwords = (aux_num_words_prev - aux_num_words_post)
            num_stopwords_removed = num_stopwords_removed + aux_num_stopwords
            if aux_num_stopwords > 0:
                num_sentences_with_stopwords += 1

        aux_sample = TreebankWordDetokenizer().detokenize(aux_sample)
        if samples.index(item)%5000 == 0:
            print(aux_sample)
        if len(aux_sample) > 0:
            samples_detokenized.append(aux_sample)
        else:
            print("item <%s> DISCARDED (0 length)!!!" % item)
    print(samples[0])
    samples = samples_detokenized

    if stopwords_removal:
        num_sentences = len(samples)
        print('[STOPWORDS REMOVAL]')
        print('num_words = %d' % num_words)
        print('num_stopwords = %d (%0.2f %%)' % (num_stopwords_removed, 100*(num_stopwords_removed/num_words)))
        print('num sentences with stopwords = %d (%0.2f %%)' % (num_sentences_with_stopwords, 100*(num_sentences_with_stopwords/num_sentences)))
        print('num_stopwords per sentence (AVG) = %0.2f' % (num_stopwords_removed/num_sentences))

f = open('./corpus/preprocessed/preprocess_%s'%corpus, 'w', encoding='utf-8')

for line in samples:
    f.write(line + '\n')
f.close()

print('Finished preprocessing corpus: %s'%corpus)

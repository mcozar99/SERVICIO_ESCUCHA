import palabrasvacias
from bertopic import BERTopic
import sklearn.metrics
import random
from subprocess import call
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer

# DO THIS COMMAND IF TRAINING DOESN´T WORK
# TORCH_HOME=<PATH_TO_ROOT> export TORCH_HOME
# EXAMPLE:
# TORCH_HOME=/home/mcozar/workspace/SERVICIO_ESCUCHA export TORCH_HOME

def sacaCorpus(corpus):
    # Devuelve una lista con todas las frases de nuestro corpus sin procesar
    print('GETTING CORPUS %s'%corpus)
    corpus = open('./corpus/%s'%corpus, 'r', encoding = 'utf-8')
    lista = []
    for line in corpus:
        lista.append(line.split('\t')[2])
    return lista

def quitaInsignificante(string):
    # Quita las palabras insignificantes de una frase en base al array de palabras vacias de significado
    cadena = []
    string = string.split(' ')
    for word in string:
        if word not in palabrasvacias.palabrasvac and 'http' not in word and word.isdigit() != True:
            cadena.append(word)
    return ' '.join(cadena)

def getSamples(corpus):
    # GETS EMBEDDING TO USE FOR TRAINING
    corpus = sacaCorpus(corpus=corpus)
    embedding = []
    i = 0
    for line in corpus:
        embedding.append(quitaInsignificante(line))
    print('EMBEDDING READY. LENGTH = %s'%embedding.__len__())
    return embedding

def randomCorpus(corpus, numero):
    # Escoge del corpus grande un corpus mas pequeño del numero que se quiera
    nuevo = []
    if numero > len(corpus):
         print('ERROR: LONGITUD ELEGIDA MAYOR QUE LONGITUD DEL CORPUS')
         return
    else:
        for i in range(numero):
            nuevo.append(corpus[random.randrange(len(corpus))])
        return nuevo

def sentence_transformer_encode(samples):
    #sentence_model = SentenceTransformer("distiluse-base-multilingual-cased-v2", device='cuda')
    sentence_model = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens', device='cuda')
    embeddings = sentence_model.encode(samples, show_progress_bar=True)
    np.savetxt("embeddings.csv", embeddings, delimiter=",")
    print('SAVED EMBEDDINGS')
    return embeddings

def defineModel(min_topic_size):
    #tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
    #sentence_model = AutoModelForMaskedLM.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
    #embeddings = sentence_model.encode(samples, show_progress_bar=True)
    #sentence_model = SentenceTransformer("dccuchile/bert-base-spanish-wwm-uncased")
    print('CREATED NEW MODEL')
    return BERTopic(language='spanish', min_topic_size=min_topic_size, verbose=True, calculate_probabilities=True)

def trainModel(model, samples, embeddings):
    print('TRAINING')
    topics, probabilities = model.fit_transform(samples, embeddings)
    print('TRAINED')
    return topics, probabilities

def saveModel(model, nombre):
    call('cd results && mkdir %s'%nombre, shell = True)
    model.save('%s.result'%nombre)
    # COMMENT IF USING WINDOWS
    call('mv %s.result ./results/%s'%(nombre, nombre), shell = True)
    call('mv probabilities.csv ./results/%s'%nombre, shell = True)
    call('mv topics.txt ./results/%s'%nombre, shell = True)
    call('mv embeddings.csv ./results/%s'%nombre, shell = True)
    # COMMENT IF USING LINUX
    #call(r'move %s.result .\results\%s' % (nombre, nombre), shell=True)
    #call(r'move probabilities.txt .\results\%s' % nombre, shell=True)
    print('SAVED MODEL WITH SUCCESS')

def saveResults(topics, probabilities):
    print('SAVING TOPICS AND PROBS')
    probabilities = probabilities.tolist()
    topics = pd.DataFrame(topics)
    topics.to_csv('./topics.txt', index=False, header=False)
    print('SAVED TOPICS')
    print('PROCESSING PROBS')
    my_df = pd.DataFrame(probabilities)
    my_df.to_csv('./probabilities.csv', index=False, header=False)
    print('SAVED PROBS')

def loadModel(model):
    # GETS MODEL YOU WANT
    return BERTopic.load('./results/%s/%s.result'%(model, model))

def getEmbeddings(model, format):
    embeddings = pd.read_csv('./results/%s/embeddings.csv'%model, header=None)
    if format == 'df':
        return embeddings
    if format == 'list':
        return embeddings.values.tolist()
    if format == 'numpy':
        return np.array(embeddings)

def getProbabilities(model, format):
    # GETS PROBS IN SOME FORMATS FOR FURTHER USES
    probs = pd.read_csv('./results/%s/probabilities.csv'%model, header=None)
    if format == 'df':
        return probs
    if format == 'list':
        return probs.values.tolist()
    if format == 'numpy':
        return np.array(probs)

def getTopics(model):
    # GETS TOPICS OF A MODEL
    f = open('./results/%s/topics.txt' % model, 'r', encoding='utf-8')
    return np.loadtxt(f).astype(int).tolist()


def getRelevants(embeddings):
    # RETURN INDEXES OF MOST RELEVANT SENTENCES IN EACH TOPIC
    maximos = []
    for i in range(embeddings.shape[1]):
        aux = embeddings[:, i]
        maximos.append(int(np.where(aux == np.max(aux))[0][0]))
    return maximos

def topicText(model, embeddings):
    # RETURNS TEXT TO PLOT IN THE REPRESENTATION
    topics = getTopics(model=model)
    relevants = getRelevants(embeddings=embeddings)
    text=[]
    for i in range(len(topics)):
        if i in relevants:
            text.append(str(topics[i]))
        else:
            text.append('')
    pd.DataFrame(text).to_csv('./results/%s/relevant_topics.csv'%model, header=None, index=None, encoding='utf-8')

def topicText2(model, embeddings):
    # RETURNS TEXT TO PLOT (TOP TOPIC'S WORDS)
    topics = getTopics(model=model)
    relevants = getRelevants(embeddings=embeddings)
    text = []
    modelo = loadModel(model)
    top = list(modelo.get_topic_freq().head(11)['Topic'])
    top.remove(-1)
    for i in range(len(topics)):
        if i in relevants and topics[i] in top:
            aux =[]
            for tupla in modelo.get_topic(topics[i]):
                aux.append(tupla[0])
            aux = ', '.join(aux)
            text.append(aux)
        else:
            text.append('')
    pd.DataFrame(text).to_csv('./results/%s/relevants.csv'%model, header=None, index=None, encoding='utf-8')

def train(corpus, name, min_topic_size, iterations):
    # TRAINS MODEL WITH AN ITERATIVE TRAINING
    samples = getSamples(corpus)
    embeddings = sentence_transformer_encode(samples)
    model = defineModel(min_topic_size=min_topic_size)
    topics, probabilities = trainModel(model, samples, embeddings)
    discards = model.get_topic_freq()['Count'][0]
    accuracy = (len(samples) - discards)/len(samples)
    print('FIRST %CLASSIFIED: '+'{:.2%}'.format(accuracy))
    iteration = 0
    while iteration < iterations:
        print('TRYING TO IMPROVE: TRY NUMBER %s OF %s'%((iteration+1), iterations))
        new_model = defineModel(min_topic_size=min_topic_size)
        new_topics, new_probs = trainModel(new_model, samples, embeddings)
        new_discards = new_model.get_topic_freq()['Count'][0]
        new_accuracy = (len(samples) - new_discards)/len(samples)
        if new_accuracy > accuracy:
            print('IMPROVED %CLASSIFIED: NEW %CLASSIFIED = '+ '{:.2%}'.format(new_accuracy))
            model, topics, probabilities, discards, accuracy = new_model, new_topics, new_probs, new_discards, new_accuracy
            iteration = 0
            print('UPDATED VALUES')
        else:
            iteration+=1
            print('DIDN´T IMPROVE: ACHIEVED %CLASSIFIED = '+'{:.2%}'.format(new_accuracy))
    print('FINISHED TRAINING. RESULTS')
    print('%CLASSIFIED: '+ '{:.2%}'.format(accuracy))
    print('NUMBER OF TOPICS: %s'%len(list(dict.fromkeys(topics))))
    print('DISCARDS: %s'%discards)
    saveResults(topics, probabilities)
    saveModel(model, '%s_MINTOPICSIZE_%s'%(name, min_topic_size))
    print('READY, GETTING RELEVANT FOR VISUALIZING')
    topicText('%s_MINTOPICSIZE_%s'%(name, min_topic_size), probabilities)
    topicText2('%s_MINTOPICSIZE_%s'%(name, min_topic_size), probabilities)


def getInfo(name):
    # GETS INFO OF A MODEL
    model = loadModel(name)
    n_topics = model.get_topic_freq().__len__()
    discards = model.get_topic_freq()['Count'][0]
    total = model.get_topic_freq()['Count'].to_numpy().sum()
    accuracy = (total-discards)/total
    print('MODEL NAME: %s'%name)
    print('NUMBER OF TOPICS: %s'%n_topics)
    print('TOTAL INPUTS: %s'%total)
    print('DISCARDS: %s'%discards)
    print('%CLASSIFIED: '+ '{:.2%}'.format(accuracy))

def getTopTopics(name, n):
    # GETS MOST REPRESENTATIVE WORDS OF TOP N TOPICS
    model = loadModel(name)
    topics = model.get_topic_freq().head(n)['Topic']
    number = model.get_topic_freq().head(n)['Count']
    for i in range(len(topics)):
        print('TOPIC: %s \t NUMBER OF INPUTS: %s'%(topics[i], number[i]))
        aux = []
        for tupla in model.get_topic(topics[i]):
            model.get_topic(topics[i])
            aux.append(tupla[0])
        aux = ', '.join(aux)
        print(aux)



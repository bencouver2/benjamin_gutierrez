"""
  Benjamin Gutierrez Garcia bencouver@gmail.com
  Topic Modeling with LDA and NMF algorithms on the ABC News Headlines Dataset
  Basado en un proyecto en linea de Coursera 
  Mining for Topics ABC and all news datasets
  using LDA = Latent Dirichlet Allocation Modeling
"""

import string
from gensim import corpora
import sklearn;
import numpy as np;
import pandas as pd;
import scipy as sp;
import sys;
#En 2.7 aun se llama pickle
import pickle;
from nltk.corpus import stopwords;
import nltk;
from gensim.models import ldamodel
import gensim.corpora;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.preprocessing import normalize;

"""
Ingestion de Datos
"""
 
#Por lo menos el dataset de ABC news esta bastante sucio, el de all news esta mejor.
#Agregamos el error_bad_lines=False para eliminar esas lineas

#The available data 
#for debugging
#data = pd.read_csv('data/small_articles1.csv', error_bad_lines=False);
#For ok
data = pd.read_csv('data/articles1.csv', error_bad_lines=False);

#Pick the column we need, headlines (ABC) or title (all news articles1)
#text_tomine = data[['headline_text']];
#For allnews, the field of the column is
text_tomine = data[['title']];

#First we need to clean up the text, removing stopwords and translating values to
#floats to be able to iterate, as we did with the categorical label data in the CIFAR100 project

text_tomine = text_tomine.astype('str');

for idx in range(len(text_tomine)):
    
    #go through each word in each text_tomine row, remove stopwords, and set them on the index.
    text_tomine.iloc[idx]['title'] = [word for word in text_tomine.iloc[idx]['title'].split(' ') if word not in stopwords.words()];
    #print logs to monitor output
    sys.stdout.write('\rc = ' + str(idx) + ' / ' + str(len(text_tomine)));

#Save the data if the dataset is too big, like the
#complete ABC news one million lines. This is done since the above cleanup
#takes ages for the ABC data
#pickle.dump(text_tomine, open('text_tomine.dat', 'wb'))

training_docset = [value[0] for value in text_tomine.iloc[0:].values];
num_topics = 10;
"""
Next step is to create an object for LDA model and train it on Document-Term matrix. 
The gensim module allows both LDA model estimation from a training corpus and inference 
of topic distribution on new, unseen documents.
https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/
"""
id2word = gensim.corpora.Dictionary(training_docset);

corpus = [id2word.doc2bow(text) for text in training_docset];

lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics);
print("")
#Print topics, words and weights, top ten words of each topic
for i in range(num_topics):
        words = lda.print_topic(i, topn = 10);
        print(words)



#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 14:30:47 2017

@author: smritijain
"""
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle

import sys
print (sys.argv)
trainpath = '../corpus/train'
validpath = '../corpus/valid'
'''
try:
    trainpath = sys.argv[1]
    validpath = sys.argv[2]
except IndexError:
        print("Please enter train dataset path followed by valid dataset path")
        sys.exit(1)
'''
train = sklearn.datasets.load_files(trainpath,categories={'book','movie'},encoding='utf-8')
valid = sklearn.datasets.load_files(validpath,categories={'book','movie'},encoding='utf-8')

#labels: train.target
#train text: train.data



#Feature Extraction with preprocessing:
    
    
# Preprocessing: tokenization, stop words removal
#tokenization: tokens of 2 or more alphanumeric characters, punctuations ignored
#converted to lowercase
#english stop words removal
#stemming


#feature extraction


stemmer = EnglishStemmer()
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


#number of occurences of words in a document
#As longer documents will have more occurences as compared to shorter document,
#Term frequencies is used - tf=occurence/length of document
#I used tf-idf score which reduces the weight of those words which occur in majory of reviews.

#Naive bayes classifier----------------------------------------------------------------------------

text_clf = Pipeline([('vect', CountVectorizer(analyzer=stemmed_words,encoding='utf-8',
                                              stop_words={'english'},lowercase='True')),
    ('tfidf', TfidfTransformer(use_idf='True')),('clf', MultinomialNB()),])

    
#parameters to be tuned here: ngram_range : tuple (min_n, max_n)The lower and upper boundary of the range of n-values for different n-grams to be extracted    
#1 gram is enough
#max_df and min_df can be tuned.
#use_idf Enable inverse-document-frequency reweighting.

parameters = {'tfidf__use_idf': (True, False),'clf__alpha': (1e-2, 1e-3),} 
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(train.data, train.target)
#Best parameters
print(gs_clf.best_params_)                                                 
predicted = gs_clf.predict(valid.data)
detailed_naive = metrics.classification_report(valid.target, predicted,target_names=valid.target_names)
acc_naive = np.mean(predicted == valid.target) 
# Validation Set
filename = '../models/naive_bayes.sav'
pickle.dump(gs_clf, open(filename, 'wb'))

print('Naive bayes accuracy on validation dataset: '+str(acc_naive))
print('Naive Bayes on validation dataset[detailed]: \n')
print(detailed_naive)

print('naive bayes completed')

del gs_clf
del predicted
#----------------------------------------------------------------------------------------------------------

#SVM-------------------------------------------------------------------------------------------------------
text_clf = Pipeline([('vect', CountVectorizer(analyzer=stemmed_words,encoding='utf-8',
                                              stop_words={'english'},lowercase='True')),
    ('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, n_iter=5, random_state=42,shuffle=True))])
parameters = {'tfidf__use_idf': (True, False),
            'clf__alpha': (1e-2, 1e-3),'clf__n_iter':[5,20],'clf__loss':('hinge','log','modified_huber','perceptron') }

#Grid Search is timeconsuming
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(train.data, train.target)
#Best parameters
print(gs_clf.best_params_)                                                 

    
predicted = gs_clf.predict(valid.data)
detailed_svm = metrics.classification_report(valid.target, predicted,target_names=valid.target_names)
acc_svm = np.mean(predicted == valid.target) 
filename = '../models/svm.sav'
pickle.dump(gs_clf, open(filename, 'wb'))

del gs_clf
del predicted

print('SVM accuracy on validation dataset: '+str(acc_svm))
print('SVM on validation dataset[detailed]: \n')
print(detailed_svm)

#----------------------------------------------------------------------------------------------------------

#Neural Network-------------------------------------------------------------------------------------------------------

text_clf = Pipeline([('vect', CountVectorizer(analyzer=stemmed_words,encoding='utf-8',
                                              stop_words={'english'},lowercase='True')),
    ('tfidf', TfidfTransformer(use_idf='True')),('clf', MLPClassifier(solver='adam', 
    alpha=1e-5, max_iter=50,hidden_layer_sizes=(10,5),random_state=1))])
#parameters = {'tfidf__use_idf': (True, False),'clf__max_iter':(20,50),'clf__hidden_layer_sizes':((10,5),(20,10),(100,50)}
#parameters = {'clf__max_iter':[4],'clf__hidden_layer_sizes':(10,5)}

#Grid Search is timeconsuming
#gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = text_clf.fit(train.data, train.target)
#Best parameters

    
predicted = gs_clf.predict(valid.data)
detailed_mlp = metrics.classification_report(valid.target, predicted,target_names=valid.target_names)
acc_mlp = np.mean(predicted == valid.target) 
filename = '../models/mp.sav'
pickle.dump(gs_clf, open(filename, 'wb'))

del gs_clf
del predicted

print('MLP accuracy on validation dataset: '+str(acc_mlp))
print('MLP on validation dataset[detailed]: \n')
print(detailed_mlp)

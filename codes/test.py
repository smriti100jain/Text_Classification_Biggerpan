#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 18:02:40 2017

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

stemmer = EnglishStemmer()

target = ['Book','Movie']
    
inp = raw_input('Enter 1. Naive bayes | 2. SVM | 3. MLP \n')

# load the model from disk
if(int(inp)==1):
    filename = '../models/naive_bayes.sav'
    text_clf = pickle.load(open(filename, 'rb'))
elif(int(inp)==2):
    filename = '../models/svm.sav'
    text_clf = pickle.load(open(filename, 'rb'))
elif(int(inp)==3):
    filename = '../models/mp.sav'
    text_clf = pickle.load(open(filename, 'rb'))

    
testdata = raw_input('Enter 1: if test corpus | Enter:2 if test input is given by user \n')
print(testdata)
if(int(testdata)==1):
    testpath = raw_input('Enter the relative path for test corpus \n')            
    test = sklearn.datasets.load_files(testpath,categories={'book','movie'},encoding='utf-8')
    predicted = text_clf.predict(test.data)
    detailed = metrics.classification_report(test.target, predicted,target_names=test.target_names)
    acc = np.mean(predicted == test.target) 
    print('detailed evaluation:'+'\n')
    print(detailed)
    print('Accuracy:'+'\n')
    print(acc)

elif(int(testdata)==2):
    testinp = raw_input('Enter the test input \n')
    testinp = [unicode(testinp,'utf-8')]
    out = text_clf.predict(testinp)
    print(target[out[0]])


        
            
    

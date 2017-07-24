#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 14:50:32 2017

@author: smritijain
"""

'''

Path: ../Corpus
Train: 600*4 = 2400 reviews [1200 reviews per class]
Test: 200*4 = 800 reviews [200 reviews per class]
Valid : 200*4 = 800 reviews [200 reviews per class]
'''
import codecs
import json
from pprint import pprint
import os
from lxml import etree

#Book reviews

      
doc = etree.parse('../book/positive.xml')


memoryElem = doc.findall('review_text')
print(len(memoryElem))
for i in range(600):
    f = codecs.open('../corpus/train/book/'+'pos'+str(i)+'.txt', "w", "utf-8")
    f.write(memoryElem[i].text)    

for i in range(600,800):
    f = codecs.open('../corpus/valid/book/'+'pos'+str(i)+'.txt', "w", "utf-8")
    f.write(memoryElem[i].text)    

for i in range(800,1000):
    f = codecs.open('../corpus/test/book/'+'pos'+str(i)+'.txt', "w", "utf-8")
    f.write(memoryElem[i].text)

doc = etree.parse('../book/negative.xml')


memoryElem = doc.findall('review_text')

for i in range(600):
    f = codecs.open('../corpus/train/book/'+'neg'+str(i)+'.txt', "w", "utf-8")
    f.write(memoryElem[i].text)    

for i in range(600,800):
    f = codecs.open('../corpus/valid/book/'+'neg'+str(i)+'.txt', "w", "utf-8")
    f.write(memoryElem[i].text)    

for i in range(800,1000):
    f = codecs.open('../corpus/test/book/'+'neg'+str(i)+'.txt', "w", "utf-8")
    f.write(memoryElem[i].text)
    

    
#IMDB reviews

files = os.listdir('../train/pos')
for i in range(600):
    g = codecs.open('../train/pos/'+files[i],'r','utf-8')
    text = g.read()
    f = codecs.open('../corpus/train/movie/'+'pos'+str(i)+'.txt', "w", "utf-8")
    f.write(text)    

for i in range(600,800):
    g = codecs.open('../train/pos/'+files[i],'r','utf-8')
    text = g.read()
    f = codecs.open('../corpus/valid/movie/'+'pos'+str(i)+'.txt', "w", "utf-8")
    f.write(text)    

for i in range(800,1000):
    g = codecs.open('../train/pos/'+files[i],'r','utf-8')
    text = g.read()
    f = codecs.open('../corpus/test/movie/'+'pos'+str(i)+'.txt', "w", "utf-8")
    f.write(text)    

files = os.listdir('../train/neg')
for i in range(600):
    g = codecs.open('../train/neg/'+files[i],'r','utf-8')
    text = g.read()
    f = codecs.open('../corpus/train/movie/'+'neg'+str(i)+'.txt', "w", "utf-8")
    f.write(text)    

for i in range(600,800):
    g = codecs.open('../train/neg/'+files[i],'r','utf-8')
    text = g.read()
    f = codecs.open('../corpus/valid/movie/'+'neg'+str(i)+'.txt', "w", "utf-8")
    f.write(text)    

for i in range(800,1000):
    g = codecs.open('../train/neg/'+files[i],'r','utf-8')
    text = g.read()
    f = codecs.open('../corpus/test/movie/'+'neg'+str(i)+'.txt', "w", "utf-8")
    f.write(text)    


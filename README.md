# Text_Classification_Biggerpan
Text classifier between book review and movie review
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 22:54:07 2017

@author: smritijain
"""

to run:
1.

python train.py ../corpus/train ../corpus/valid

Models are saved in ../models

2. In order to test saved models

python test.py

Following inputs will be asked:

(i) Which model to load
Enter 1. Naive bayes | 2. SVM | 3. MLP 
> 1

(ii) Whether to test on test corpus OR test input is entered by user
Enter 1: if test corpus | Enter:2 if test input is given by user
> 1

(iii) if test data is taken from corpus 
'Enter the relative path for test corpus 
> ../corpus/test

(iv) if test data entered by user:
Enter the test input
> It was very engaging. The actors did a really great job.

Output: Movie

           

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 12:04:16 2015

@author: BL4685
"""
import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("C:/move - bwlee/Data Analysis/Nano/Intro to Machine Learning/1/ud120-projects/tools/")
sys.path.append("C:/move - bwlee/Data Analysis/Nano/Intro to Machine Learning/1/ud120-projects/final_project")
import math
import itertools
import logging
import scrapy
import twisted
from twisted.python import log

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from itertools import *
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

cv = StratifiedShuffleSplit(targets, 10, random_state = 42)

parameters = [{'select__k': [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
               'select__score_func': [f_classif, chi2],
               'classifier__C': [1, 10, 100, 1000],
              }]

pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('select', SelectKBest()),
        ("classifier", LinearSVC())
    ])

scores = ["accuracy", 'precision',"recall"]
clf={}
results = {}

for score in scores:
    print("Tuning hyper-parameters for %s" % score)
    print()
    
    clf[score] = GridSearchCV(pipeline, parameters, cv = cv, scoring= score)
    clf[score].fit(features, targets)
    print clf[score].best_params_
    
    #pass the best_estimator from gridsearch to the test_classifier
    test_classifier(clf[score].best_estimator_, enron_data, financial_features)
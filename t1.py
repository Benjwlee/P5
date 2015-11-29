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
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#Setting up kfold validation with 4 folds
kf_total = cross_validation.KFold(len(features), n_folds=4,
                                  indices=True, shuffle=True,
                                  random_state=4)
#Creating arrays for parameters to test with gridsearchCV
C_range = np.logspace(-2,5,8)
gamma_range = np.logspace(-5,2,8)
#setting up gridsearchCV
parameters = {'kernel':('linear','rbf'),'C':C_range, 'gamma':gamma_range}
svr = svm.SVC()
if __name__ == "__main__":
    clf2 = grid_search.GridSearchCV(svr, parameters, verbose=5, n_jobs=-1)
    accuracy=[]
    for train_index, test_index in kf_total:
        features_train = [features[i] for i in train_index]
        features_test = [features[i] for i in test_index]
        labels_train = [labels[i] for i in train_index]
        labels_test = [labels[i] for i in test_index]
        clf2.fit(features_train, labels_train)
        pred = clf2.predict(features_test)
        acc = accuracy_score(pred, labels_test)
        accuracy.append(acc)
        print "BEST ESTIMATOR: ", clf2.best_estimator_
    print accuracy
    print np.mean(accuracy)
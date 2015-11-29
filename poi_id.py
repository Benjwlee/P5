#!/usr/bin/python

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

def happy():
  #for name in sorted(my_dataset.keys()):
  #  for fld in sorted(my_dataset[name].keys()):
  #    if my_dataset[name][fld] == 'NaN':
  #      my_dataset[name][fld] =0
  #      zerocnt[fld]=zerocnt.get(fld,0)+1
  #for name in sorted(my_dataset.keys()):
  ##    my_dataset[name]['incsum']=math.log(my_dataset[name]['salary']+my_dataset[name]['bonus']+1, 10)
  #    my_dataset[name]['incsum']=my_dataset[name]['salary']+my_dataset[name]['bonus']\
  #        +my_dataset[name]['exercised_stock_options']+my_dataset[name]['restricted_stock']+1
  ##    my_dataset[name]['incema']=my_dataset[name]['exercised_stock_options']+my_dataset[name]['restricted_stock']
  #    my_dataset[name]['incema']=my_dataset[name]['from_poi_to_this_person']\
  #        +my_dataset[name]['from_this_person_to_poi']+1
  #        #+my_dataset[name]['shared_receipt_with_poi']
  ##    my_dataset[name]['incema']=math.log(my_dataset[name]['from_poi_to_this_person']\
  ##        +my_dataset[name]['from_this_person_to_poi']+1, 2)
  ##    #    /(my_dataset[name]['from_messages']+my_dataset[name]['to_messages'])
  #print zerocnt
  #incsum_poi=[]
  #incema_poi=[]
  #incsum_npoi=[]
  #incema_npoi=[]
  #print "There are " + str(len(my_dataset['TOTAL'])) + " features."
  #poicnt=0
  #npoicnt=0
  #f1=open('./testfile', 'w+')
  #for k,v in (data_dict).iteritems() :
  #    print >>f1, k,v
  #my_dataset.pop('TOTAL', 0)
  #for d in my_dataset:
  #    if my_dataset[d]['poi'] == True:
  #        poicnt+=1
  #        incsum_poi.append(my_dataset[d]['incsum'])
  #        incema_poi.append(my_dataset[d]['incema'])
  #    elif my_dataset[d]['poi'] == False:
  #        npoicnt+=1
  #        incsum_npoi.append(my_dataset[d]['incsum'])
  #        incema_npoi.append(my_dataset[d]['incema'])
  #print str(poicnt) + " poi, "+str(npoicnt) + " none poi in data"
  #
  #plt.scatter(incsum_poi, incema_poi, color = "b", label="poi")
  #plt.scatter(incsum_npoi, incema_npoi, color = "r", label="not poi")
  #plt.legend()
  #plt.xlabel("income combined")
  #plt.ylabel("email combined")
  #plt.show()
  
  ### Extract features and labels from dataset for local testing
  
  data = featureFormat(my_dataset, features_list, sort_keys = True)
  labels, features = targetFeatureSplit(data)
  #sys.exit()
  
  ### Task 4: Try a varity of classifiers
  ### Please name your classifier clf for easy export below.
  ### Note that if you want to do PCA or other multi-stage operations,
  ### you'll need to use Pipelines. For more info:
  ### http://scikit-learn.org/stable/modules/pipeline.html

  classi = [
    KMeans(n_clusters=3, n_init=1, init='random'),
    KMeans(n_clusters=5, n_init=1, init='random'),
    LinearSVC(C=1.0),
    #SVC(gamma=3, C=1),
    #SVC(gamma=3, C=2),
    #SVC(gamma=4, C=1),
    #svm.SVC(kernel='rbf'),
    RandomForestClassifier(n_estimators=100),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    RandomForestClassifier(max_depth=10, n_estimators=15, max_features=2),
    KNeighborsClassifier(2),
    KNeighborsClassifier(1),
    KNeighborsClassifier(5),
    DecisionTreeClassifier(max_depth=5, criterion='gini'),
    DecisionTreeClassifier(max_depth=5, criterion='entropy'),
    DecisionTreeClassifier(max_depth=3),
    AdaBoostClassifier(),
    AdaBoostClassifier(algorithm='SAMME'),
    AdaBoostClassifier(algorithm='SAMME',learning_rate=0.5,n_estimators=100),
    AdaBoostClassifier(algorithm='SAMME',n_estimators=100),
    GaussianNB(),
    LDA(),
    QDA(),
    LogisticRegression()]
  for cl in classi:
    test_classifier(cl, my_dataset, features_list)
  #lr = LogisticRegression()
  ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
  ### using our testing script.
  ### Because of the small size of the dataset, the script uses stratified
  ### shuffle split cross validation. For more info: 
  ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
  #test_classifier(clf, my_dataset, features_list)
  ### Dump your classifier, dataset, and features_list so 
  ### anyone can run/check your results.
  clf=KMeans(n_clusters=5, n_init=1, init='random')
  test_classifier(clf, my_dataset, features_list)
  dump_classifier_and_data(clf, my_dataset, features_list)
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary', 'from_this_person_to_poi',\
# 'from_poi_to_this_person'] # You will need to use more features
#features_list = ['poi','incsum', 'incema'] # You will need to use more features
#all_features = ['poi','salary','deferral_payments','total_payments','bonus','loan_advances',
#  'other','director_fees','deferred_income','long_term_incentive','expenses','exercised_stock_options',\
#  'restricted_stock','restricted_stock_deferred','total_stock_value','to_messages','from_messages',\
#  'from_this_person_to_poi','from_poi_to_this_person','shared_receipt_with_poi','poi','email_address']
#features_list = ['poi','salary','bonus','director_fees','deferred_income','long_term_incentive','expenses','total_stock_value'] # You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
zerocnt={}
log.startLogging(open('foo.log', 'w'))

all_features = ['poi','salary','total_payments','bonus','loan_advances']
for L in range(4, len(all_features)+1):
        for features_list in itertools.permutations(all_features, L):
                print('<<<<'+','.join(features_list)+'>>>>>>>')
                happy()
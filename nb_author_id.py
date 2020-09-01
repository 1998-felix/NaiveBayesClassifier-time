#!/usr/bin/python

"used data to classify who wrote the email"
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf = GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)

print "Training time:", round((time()-t0),3), "s"

t0 = time()
pred_y = clf.predict(features_test)
print "Predicting time:", round((time()-t0),3), "s"

score = accuracy_score(labels_test, pred_y)

print score



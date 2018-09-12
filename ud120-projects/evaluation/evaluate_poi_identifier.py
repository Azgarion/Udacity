#!/usr/bin/python

import pickle
import sys

sys.path.append("../tools/")

from time import time
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from feature_format import featureFormat, targetFeatureSplit

"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

data_dict = pickle.load(
    open("../final_project/final_project_dataset.pkl", "r"))

# add more features to features_list!
features_list = ["poi", "salary", "bonus"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

# your code goes here

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels,
    test_size=0.3, random_state=42)

# it's all yours from here forward!   = preprocess()

clf = tree.DecisionTreeClassifier()

t0 = time()
clf = clf.fit(features_train, labels_train)
print "training time:", round(time() - t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "training time:", round(time() - t0, 3), "s"

acc = accuracy_score(pred, labels_test)
print confusion_matrix(pred, labels_test, labels=range(2))
print acc, len(labels_test)

print "---------------------------------"

print precision_score(labels_test, pred, average='micro')
print precision_score(labels_test, pred, average='binary')
print precision_score(labels_test, pred, average='macro')

print "---------------------------------"

print recall_score(labels_test, pred, average='micro')
print recall_score(labels_test, pred, average='binary')
print recall_score(labels_test, pred, average='macro')

print "---------------------------------"

# for i, elt in enumerate(pred):
#         print pred[i], labels_test[i]

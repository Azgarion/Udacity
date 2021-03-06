#!/usr/bin/python

import pickle
import numpy as np
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier

np.random.seed(42)

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# The words (features) and authors (labels), already largely processed.
# These files should have been created from the previous (Lesson 10)
#  mini-project.
words_file = "../text_learning/your_word_data.pkl"
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load(open(words_file, "r"))
authors = pickle.load(open(authors_file, "r"))

# test_size is the percentage of events assigned to the test set (the
# remainder go into training)
# feature matrices changed to dense representations for compatibility with
# classifier functions in versions 0.15.2 and earlier

features_train, features_test, labels_train, labels_test =\
    cross_validation.train_test_split(
        word_data, authors,
        test_size=0.1, random_state=42)

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test).toarray()

g = vectorizer.get_feature_names()

print g[33698]
print g[14349]
print g[32714]

# a classic way to overfit is to use a small number
# of data points and a large number of features;
# train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train = labels_train[:150]

# your code goes here

clf = DecisionTreeClassifier()

clf.fit(features_train, labels_train)

print clf.score(features_test, labels_test)

importances = clf.feature_importances_

print [(i, importance) for i, importance
       in enumerate(importances) if importance > 0.1]

# std = np.std([clf.feature_importances_ for tree in clf.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]

print len(features_train)

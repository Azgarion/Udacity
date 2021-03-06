#!/usr/bin/python

"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""    
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from feature_format import featureFormat, targetFeatureSplit

dictionary = pickle.load(
    open("../final_project/final_project_dataset_modified.pkl", "r"))

# list the features you want to look at--first item in the
# list will be the "target" feature
features_list = ["bonus", "salary"]
data = featureFormat(dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit(data)
# training-testing split needed in regression, just like classification

feature_train, feature_test,\
    target_train, target_test = train_test_split(
        features, target,
        test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"

# Your regression goes here!
# Please name it reg, so that the plotting code below picks it up and
# plots it correctly. Don't forget to change the test_color above from "b" to
# "r" to differentiate training points from test points.

reg = linear_model.LinearRegression()
reg.fit(feature_train, target_train)
print reg.coef_
print reg.intercept_


pred = reg.predict(feature_test)
# score =  reg.score(feature_train,target_train)
# print "score: ",score
print("Mean squared error:",
      mean_squared_error(pred, target_test))
# Explained variance score: 1 is perfect prediction
print('Variance score: ', r2_score(target_test, pred))

# draw the scatterplot, with color-coded training and testing points
for feature, target in zip(feature_test, target_test):
    plt.scatter(feature, target, color=test_color)
for feature, target in zip(feature_train, target_train):
    plt.scatter(feature, target, color=train_color)

# labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")

# draw the regression line, once it's coded

try:
    plt.plot(feature_test, reg.predict(feature_test))
except NameError:
    pass
reg.fit(feature_test, target_test)
print reg.coef_
pred = reg.predict(feature_train)

score = reg.score(feature_train, target_train)
print "score: ", score
plt.plot(feature_train, reg.predict(feature_train), color="g")


plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()

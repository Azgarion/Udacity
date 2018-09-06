#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from time import time
from class_vis import prettyPicture
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]



#### initial visualization
plt.subplot(361)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
# plt.show()

################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary


#################################################################################


# KNN Algorithmes

# 
# 
# 
# 
#



from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

row = 3
col = 6
num = 2

errors = []

lowest_error = 0.0
highest_error = 1.0

worst_clf = 0
best_clf = 0

name_best = ""
name_worst = ""

for k in range(2,15) :
	
	num += 1

	name =  "k=" + str(k)  
	print name
	clf = KNeighborsClassifier(n_neighbors=k)
	
	fit_t0 = time()
	clf.fit( features_train,labels_train)
	# print "training time:", round(time()-fit_t0, 3), "s"
	

	pred_t0 = time()
	pred = clf.predict( features_test )
	# print "test time:", round(time()-pred_t0, 3), "s"

	acc = accuracy_score(pred,labels_test)
	
	if acc > lowest_error :
		best_clf = clf
		lowest_error = acc
		name_best = name

	elif highest_error > acc  :
		worst_clf = clf
		highest_error = acc 
		name_worst = name

	
	try:
		prettyPicture(clf, features_test, labels_test, row, col, num, name)

	except NameError:
		print "coucou"
	

	errors.append(100*(1-acc))


plt.subplot(362)
plt.plot(range(2,15), errors, 'o-')
plt.show()

print lowest_error, "hello"

# 
# 
# 
# 
# 

#################################################################################

# # Adaboost

# from sklearn.model_selection import cross_val_score
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn import svm
# from sklearn.metrics import accuracy_score

# n_estim = 0

# row = 3
# col = 6
# num = 1

# lowest_error = 0.0
# highest_error = 1.0

# worst_clf = 0
# best_clf = 0

# name_best = ""
# name_worst = ""

# param=[]

# for elt in param:


# 	n_estim += 250 
# 	name =  "k=" + str(k)  
# 	clf = AdaBoostClassifier(n_estimators=n_estim)

# 	print clf , "hello"
# 	scores = cross_val_score(clf, features_train, labels_train)
# 	print "hi", scores
# 	clf.fit(features_train,labels_train)

# 	pred = clf.predict( features_test )

# 	# 	# print "test time:", round(time()-pred_t0, 3), "s"
# 	acc = accuracy_score(pred,labels_test)
# 	print acc
	
# 	if acc > lowest_error :
# 		best_clf = clf
# 		lowest_error = acc
# 		name_best = name

# 	elif highest_error > acc  :
# 		worst_clf = clf
# 		highest_error = acc 
# 		name_worst = name

try:
	num +=1
	name = "best_clf " + name_best
	prettyPicture(best_clf, features_test, labels_test, row, col, num, name)
	num +=1
	name = " worst_clf " + name_worst
	prettyPicture(worst_clf, features_test, labels_test, row, col, num, name)
except NameError:
	print "coucou"
plt.show()


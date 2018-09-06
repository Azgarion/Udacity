#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""

"""HELLO WORLD """


import pickle
import numpy
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler



def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)
min_h = 100000000
max_h = 0
people = 0 
h_key = {}
scaler = MinMaxScaler()
for key,valeur in data_dict.items():
    # print enron_data['{}'.format (key) ]['salary']
    people += 1
    h = data_dict['{}'.format (key) ]['from_messages']
    print key, "/  messages: " , h
    if h > max_h and h != 'NaN': 
        max_h = h 
        h_key = key , valeur
    elif h < min_h and h != 'NaN':
        min_h = h

print "MIN: ", min_h, "MAX: ", max_h, "PEOPLE: ", people 
# print h_keyprint "MIN: ", min_h, "MAX: ", max_h, "PEOPLE: ", people 

### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
max_h = 0
max_h2 =0 
min_h2 =10000000
scaler.fit(finance_features)
print finance_features[0:5]
print scaler.transform([[200000., 1000000.]])
scaled_features = scaler.transform(finance_features)
print 
for f1, f2 in finance_features:
    
    print "SALARY: ", f1 ,"/////STOCK: ", f2
    if f1 > max_h and f1 != 'NaN': 
        max_h = f1
    elif f1 < min_h and f1 != 'NaN' and f1!= 0:
        min_h = f1

    if f2 > max_h2 and f2 != 'NaN': 
        max_h2 = f2
    elif f2 < min_h2 and f2 != 'NaN' and f2!= 0:
        min_h2 = f2

    plt.scatter( f1, f2 )
plt.show()
print "MIN: ", min_h, "MAX: ", max_h, "PEOPLE: ", people 
print "MIN: ", min_h2, "MAX: ", max_h2
### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred


kmeans = KMeans(n_clusters=2).fit(finance_features)
print kmeans.labels_
pred = kmeans.predict(finance_features)



### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"

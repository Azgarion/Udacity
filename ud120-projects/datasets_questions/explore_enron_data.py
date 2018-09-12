#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import math

enron_data = pickle.load(
    open("../final_project/final_project_dataset.pkl", "r"))
enron_data.pop("TOTAL", 0)
count = 0
mail = 0
payments_missing = 0
people = 0

min_h = 0
max_h = 0

for key, valeur in enron_data.items():
    # print enron_data['{}'.format (key) ]['salary']
    people += 1
    h = enron_data['{}'.format(key)]['exercised_stock_options']
    print key, h
    if h > max_h and (enron_data['{}'.format(key)]
                      ['exercised_stock_options'] != 'NaN'):
            max_h = h
    elif h < min_h and h != 'NaN':
        min_h = h

    # if not key.pop("email_address") == 'NaN':
    # 	mail = mail +1
    # if key.pop("poi") and key.pop('total_payments') == 'NaN':
    # payments_missing += 1

print "MIN: ", min_h, "MAX: ", max_h
# print sorted(enron_data.keys())

# 	# 'poi' in elt.values()
# print "total_salary:
# ", count,"total_mail: ", mail,"total missing payments: ",payments_missing
# print people

# print enron_data['TOTAL']
# print enron_data['FASTOW ANDREW S']['total_payments']
# print enron_data['SKILLING JEFFREY K']['total_payments']

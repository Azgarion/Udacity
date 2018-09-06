#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    # print predictions
    # print ages
    # print net_worths
    from operator import itemgetter

    cleaned_data = []
    errors = []
    
    for i, elt in enumerate(predictions):

        errors.insert(i, predictions[i]-net_worths[i])
        cleaned_data.insert(i, ( ages[i], net_worths[i], errors[i] ) )


        # print i, ( ages[i], net_worths[i], errors[i] )
    cleaned_data =sorted(cleaned_data, key=itemgetter((2)))
    errors.sort()
    # cleaned_data.sort(key=lambda tup: tup[2])
    print len(cleaned_data)*0.9
    # print ages[5]   
    # print cleaned_data[50]
    # predictions , ages, net_worths = predictions[:len(predictions)*0.9], ages[:len(ages)*0.9], net_worths[:len(net_worths)*0.9]
    cleaned_data = cleaned_data[:int(len(cleaned_data)*0.9)]
    # for elt in cleaned_data :
    #     print elt[2]
    ### your code goes here

    
    return cleaned_data


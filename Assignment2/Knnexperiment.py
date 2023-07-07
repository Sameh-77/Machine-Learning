#======================================================#
#== Sameh Algharabli - CNG 409 -- Assignment2 -- KNNExperiment ==#
#======================================================#
#--- Libraries ---#
import pickle
from Distance import Distance
from Knn import KNN
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
import math as m
#-------------------------------------------------------#
# Reading the data and the labels #
dataset, labels = pickle.load(open("../data/part1_dataset.data", "rb"))
#-------------------------------------------------------#
""""
apply_cross fold function is responsible for doing the cross fold 
validation with n_splits = 5 and n_repeats = 5
inside this function we create the classifier and get the predictions 
as well as the accuracy 
"""
def apply_crossfold(dataset, labels, metric, similarity_function_parameters, k):
    k_accuracies = []
    kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)

    for train_indices, test_indices in kfold.split(dataset, labels):

        # train data depending on the splits
        current_train = dataset[train_indices]
        # train labels depending on the splits
        current_train_label = labels[train_indices]

        # creating a KNN classifier with its parameters
        KNNclassifier = KNN(current_train, current_train_label, metric, similarity_function_parameters, k)

        # test data depending on the splits
        current_test = dataset[test_indices]
        # test labels depedning on the splits
        current_test_labels = labels[test_indices]

        # getting predictions of the test data
        predicted = KNNclassifier.predict(current_test)

        # evaluating the predictions of the test data with the original test labels
        accuracy = KNNclassifier.evaluate(current_test_labels, predicted)

        k_accuracies.append(accuracy)
    return k_accuracies
#---------------------------------------#


similarity_functions = ["cosine", "minkowski","mahalanobis"]
k_values = [5, 10, 30]

results_list = []
""" 
Nested loops to do the grid search, I have 9 configurations 
"""
for metric in similarity_functions:
    for k in k_values:
        print("Running for k = {} and Metric = {}".format(k, metric))

        # I'm taking all the accuracies after all the repeats of the cross validation
        k_accuracies = apply_crossfold(dataset, labels, metric, None, k)

        # finding the mean and the confidence interval of the accuracy for the configuration
        mean_accuracy = sum(k_accuracies)/len(k_accuracies)
        standard_deviation = np.std(k_accuracies)  # calculating standard_deviation
        confidence_interval_min = mean_accuracy - (1.96 * standard_deviation / m.sqrt(len(k_accuracies)))  # finding min in confidence_interval
        confidence_interval_max = mean_accuracy + (1.96 * standard_deviation / m.sqrt(len(k_accuracies)))  # finding max in confidence_interval

        # a dictionary that stores the hyperparameters for each configuration along with its mean accuracy
        result_dict = {"k": k, "mean_accuracy": mean_accuracy, "metric": metric,"Accuracy Confidence Interval": [confidence_interval_min,confidence_interval_max]}
        print("Mean Accuracy for k = {}, Metric = {} is {}, Confidence Interval = {}".format(k, metric, mean_accuracy,[confidence_interval_min,confidence_interval_max]))
        print("==============================================================================")
        results_list.append(result_dict)
#-----------------------------
# Finding the best highest mean value and the best parameters #
best_k = 0
best_metric = 0
max_accuracy = 0
for parameters in results_list:
    if parameters['mean_accuracy']>max_accuracy:
        max_accuracy=parameters['mean_accuracy']
        best_metric = parameters['metric']
        best_k = parameters['k']
print("#--------------------------------------#")
print("The best parameters that scored the highest accuracy are: ")
print("Best Accuracy: ", max_accuracy)
print("Best_k: ", best_k)
print("Best_metric: ", best_metric)
#---------------------------

# I do not have a testing set :)
# print("#--------------------------------------#")
# print("Buiding the KNN with the best parameters with the full dataset...")
#
# #building the nnew knn with the best parameters
# KNNclassifier = KNN(dataset, labels, similarity_function=best_metric,similarity_function_parameters=similarity_function_parameters, K=best_k)
#
# predicted = KNNclassifier.predict(testset)
# accuracy = accuracy_score(test_labels, predicted)

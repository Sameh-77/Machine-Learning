#======================================================#
#== Sameh Algharabli - CNG 409 -- Assignment3 --  ==#
#======================================================#
# accurfacy and f1 score
# confidence intervals

# Libraries #
from DataLoader import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import math as m
#--------------------------#
data_path = "../data/credit.data"

dataset, labels = DataLoader.load_credit_with_onehot(data_path)

import random
random.seed(11)
np.random.seed(17)

# PARAMETERS GRIDS #
knn_parameter_grid = {"kneighborsclassifier__metric": ["cosine", "euclidean"],
                          "kneighborsclassifier__n_neighbors": [2,  4]
                          }

svm_parameter_grid = {"svc__C": [0.5],
              "svc__kernel": ["poly", "rbf"]
}

tree_parameter_grid = {"decisiontreeclassifier__criterion": ["gini","entropy"],
              "decisiontreeclassifier__splitter": ["best"]
}

RF_parameter_grid = {"n_estimators": [100],
              "criterion": ["gini","entropy"]
}
#--------------------------------#

outer_cross_validation = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=np.random.randint(1, 1000))
inner_cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=np.random.randint(1, 1000))

knn_performance = []
svm_performance = []
tree_performance = []
#RF_performance = []

# This stores the overall performance for each method #
knn_overall_performance = []
svm_overall_performance = []
tree_overall_performance = []
RF_overall_performance = []

def displayConf(model,outersplit,bestConfDic):
    #print("Outersplit --> ",outersplit)
    #print("-----------------------------------")
    #print("The configurations are as follows: ")
    #print(model.cv_results_["params"], sep="\n")
    #print("-------------------------------------------------------\n")
    # for i in range(0, 10):
    #     name = "split" + str(i) + "_test_score"
    #     splitResults = list(model.cv_results_[name])
    #     print("Split {} --> {}".format(i, splitResults))
    #     print("---------")

    #print(model.cv_results_)
    # print(svmgrid.cv_results_["params"])
    # mean_accuracie= model.cv_results_["mean_test_score"]
    # std = model.cv_results_["std_test_score"]
    # #standard_deviation = np.std(RF_overall_performance)  # calculating standard_deviation
    # confidence_interval_min = mean_accuracie - (
    #             1.96 * std / m.sqrt(len(RF_overall_performance)))  # finding min in confidence_interval
    # confidence_interval_max = np.mean(RF_overall_performance) + (
    #             1.96 * standard_deviation / m.sqrt(len(RF_overall_performance)))  # finding max in confidence_interval
    best_score = model.best_score_
    best_Conf = model.best_params_
    value = (best_score,best_Conf)
    name = "split " + str(outersplit)
    bestConfDic[name] = value

    #print("Best mean accuracy: ", model.best_score_)

    #print("END OF OUTERSPLIT ",outersplit)
    #print("------------------------------------------------------------")
    return bestConfDic
i = 0
bestKNNCong = {} # this store the best configuration after each outer split for KNN
bestSVMCong = {} # this store the best configuration after each outer split for SVM
bestTreeCong = {} # this store the best configuration after each outer split for DT
bestRFCong = {} # this store the best configuration after each outer split for RF

# outer cross validation loop
for train_indices, test_indices in outer_cross_validation.split(dataset, labels):
    current_training_part = dataset[train_indices] # taking the training part
    current_training_part_RF = dataset[train_indices].copy() # a copy is made to be used for the RF

    current_training_part_label = labels[train_indices] # this is taking the training labels
    current_training_part_label_RF = labels[train_indices].copy() # a copy is made to be used for the RF


    #------------------- KNN TRAINING----------------------------#
    knn_pipeline = make_pipeline(MinMaxScaler(), KNeighborsClassifier())
    knn_grid_search = GridSearchCV(knn_pipeline, param_grid=knn_parameter_grid, refit=True, cv=inner_cross_validation, scoring="f1")
    knn_grid_search.fit(current_training_part, current_training_part_label)
    bestKNNCong = displayConf(knn_grid_search, i, bestKNNCong)  # displaying KNN RESULTS

    #--------------SVM TRAINING---------------------------------#
    svm_pipeline = make_pipeline(MinMaxScaler(), SVC())
    svm_grid_search = GridSearchCV(svm_pipeline, param_grid=svm_parameter_grid, refit=True, cv=inner_cross_validation, scoring="f1")
    svm_grid_search.fit(current_training_part, current_training_part_label)
    bestSVMCong = displayConf(svm_grid_search,i,bestSVMCong) # displaying SVM RESULTS

    # --------------DECISION TREE TRAINING---------------------------------#
    tree_pipeline = make_pipeline(MinMaxScaler(), DecisionTreeClassifier())
    tree_grid_search = GridSearchCV(tree_pipeline, param_grid=tree_parameter_grid, refit=True, cv=inner_cross_validation,scoring="f1")
    tree_grid_search.fit(current_training_part, current_training_part_label)
    bestTreeCong = displayConf(tree_grid_search, i, bestTreeCong)  # displaying tree RESULTS

    #-----------------------------------------TESTING----------------------------------------------------------#

    #---------- GETTING TESTING PART --------------------------#
    current_test_part = dataset[test_indices] # taking the testing part
    current_test_part_label = labels[test_indices] # taking the testing part labels

    current_test_part_RF = dataset[test_indices].copy() # a copy is made to be used for the RF
    current_test_part_label_RF = labels[test_indices].copy() # a copy is made to be used for the RF
    #--------- KNN PREDICTION --------------------#
    knn_predicted = knn_grid_search.predict(current_test_part)
    knn_overall_performance.append(f1_score(current_test_part_label, knn_predicted, average="micro"))

    #-----------SVM PREDICTION------------------#
    svm_predicted = svm_grid_search.predict(current_test_part)
    svm_overall_performance.append(f1_score(current_test_part_label, svm_predicted, average="micro"))

    # -----------Decision tree PREDICTION------------------#
    tree_predicted = tree_grid_search.predict(current_test_part)
    tree_overall_performance.append(f1_score(current_test_part_label, tree_predicted, average="micro"))


    #----------- RANDOM FOREST ---------------------#
    RF_performance = dict()
    # A loop for the inner cross validation
    for inner_train_indices, inner_test_indices in inner_cross_validation.split(current_training_part,
                                                                                current_training_part_label):

        inner_training_dataset = current_training_part[inner_train_indices] # taking the inner traing data
        inner_training_label = current_training_part_label[inner_train_indices] # taking the inner training data labels

        inner_test_dataset = current_training_part[inner_test_indices] # taking the inner testing data
        inner_test_label = current_training_part_label[inner_test_indices] # taking the inner testing data labels

        inner_scalar = MinMaxScaler() # applying min max
        scaled_inner_training_dataset = inner_scalar.fit_transform(inner_training_dataset)

        scaled_inner_test_dataset = inner_scalar.transform(inner_test_dataset) # transforming the normalized data

        # grid search for each hyperparameter
        for n_estimator in RF_parameter_grid["n_estimators"]:
            for criterion in RF_parameter_grid["criterion"]:
                loop_scores = []
                # repeating 5 times (fitting the model 5 times)
                for j in range(5):

                    randomForest = RandomForestClassifier(n_estimators=n_estimator, criterion=criterion)
                    randomForest.fit(scaled_inner_training_dataset, inner_training_label)

                    predicted = randomForest.predict(scaled_inner_test_dataset)
                    # adding all scores from the 5 runs to a list, and then storing the mean of them
                    loop_scores.append(f1_score(inner_test_label, predicted))

                if (n_estimator, criterion) not in RF_performance:
                    #x = (n_estimator,criterion)
                    RF_performance[(n_estimator, criterion)] = []
                RF_performance[(n_estimator, criterion)].append(np.mean(loop_scores))
    best_parameter_RF = None
    best_score_RF = -float('inf')
    # finding the best configuration
    for param_config in RF_performance:
        v = np.mean(RF_performance[param_config])
        if v > best_score_RF:
            best_score_RF = v
            best_parameter_RF = param_config

    # stroing the best configuration for each outer split and its mean value
    value = (best_score_RF,best_parameter_RF)
    name = "split " + str(i)
    bestRFCong[name] = value
    #print(best_parameter_RF)

    outer_scaler = MinMaxScaler()

    # refitting the best model
    scaled_current_training_part = outer_scaler.fit_transform(current_training_part_RF)
    RF_with_best_param = RandomForestClassifier(n_estimators=best_parameter_RF[0], criterion=best_parameter_RF[1])
    RF_with_best_param.fit(scaled_current_training_part, current_training_part_label_RF)

    scaled_current_test_part = outer_scaler.transform(current_test_part_RF)

    # calculating the performance of the best model
    RF_predicted = RF_with_best_param.predict(scaled_current_test_part)
    RF_overall_performance.append(f1_score(current_test_part_label_RF, RF_predicted))
    print("done: ",i)
    i += 1


# --------------- DISPLAYING RESULTS -----------------#
print("KNN RESULTS\n----------------")
for key in bestKNNCong.keys():
    print("{}: {}".format(key,bestKNNCong[key]))
print("KNN OVERALL PERFORMANCE: ", knn_overall_performance)
print("KNN OVERALL PERFORMANCE MEAN: ", np.mean(knn_overall_performance))
standard_deviation = np.std(knn_overall_performance)  # calculating standard_deviation
confidence_interval_min = np.mean(knn_overall_performance) - (1.96 * standard_deviation / m.sqrt(len(knn_overall_performance)))  # finding min in confidence_interval
confidence_interval_max = np.mean(knn_overall_performance) + (1.96 * standard_deviation / m.sqrt(len(knn_overall_performance)))  # finding max in confidence_interval
print("KNN OVERALL PERFORMANCE Confidence Interval: ",(confidence_interval_min,confidence_interval_max))
print("-----------------------------------------")

print("SVM RESULTS\n----------------")
for key in bestSVMCong.keys():
    print("{}: {}".format(key,bestSVMCong[key]))
print("SVM OVERALL PERFORMANCE: ",svm_overall_performance)
print("SVM OVERALL PERFORMANCE MEAN: ", np.mean(svm_overall_performance))
standard_deviation = np.std(svm_overall_performance)  # calculating standard_deviation
confidence_interval_min = np.mean(svm_overall_performance) - (1.96 * standard_deviation / m.sqrt(len(svm_overall_performance)))  # finding min in confidence_interval
confidence_interval_max = np.mean(svm_overall_performance) + (1.96 * standard_deviation / m.sqrt(len(svm_overall_performance)))  # finding max in confidence_interval
print("SVM OVERALL PERFORMANCE Confidence Interval: ",(confidence_interval_min,confidence_interval_max))
print("-----------------------------------------")

print("Decision Tree RESULTS\n----------------")
for key in bestTreeCong.keys():
    print("{}: {}".format(key,bestTreeCong[key]))
print("Decision Tree OVERALL PERFORMANCE: ",tree_overall_performance)
print("Decision Tree OVERALL PERFORMANCE MEAN: ", np.mean(tree_overall_performance))
standard_deviation = np.std(tree_overall_performance)  # calculating standard_deviation
confidence_interval_min = np.mean(tree_overall_performance) - (1.96 * standard_deviation / m.sqrt(len(tree_overall_performance)))  # finding min in confidence_interval
confidence_interval_max = np.mean(tree_overall_performance) + (1.96 * standard_deviation / m.sqrt(len(tree_overall_performance)))  # finding max in confidence_interval
print("Decision Tree OVERALL PERFORMANCE Confidence Interval: ",(confidence_interval_min,confidence_interval_max))
print("-----------------------------------------")

print("Random Forest RESULTS\n----------------")
for key in bestRFCong.keys():
    print("{}: {}".format(key,bestRFCong[key]))
print("Random Forest OVERALL PERFORMANCE: ",RF_overall_performance)
print("Random Forest OVERALL PERFORMANCE MEAN: ", np.mean(RF_overall_performance))
standard_deviation = np.std(RF_overall_performance)  # calculating standard_deviation
confidence_interval_min = np.mean(RF_overall_performance) - (1.96 * standard_deviation / m.sqrt(len(RF_overall_performance)))  # finding min in confidence_interval
confidence_interval_max = np.mean(RF_overall_performance) + (1.96 * standard_deviation / m.sqrt(len(RF_overall_performance)))  # finding max in confidence_interval
print("Random Forest OVERALL PERFORMANCE Confidence Interval: ",(confidence_interval_min,confidence_interval_max))
print("-----------------------------------------")
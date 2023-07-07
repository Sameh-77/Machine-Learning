#======================================================#
#== Sameh Algharabli - CNG 409 -- Assignment3 --  ==#
#======================================================#

import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import math as m

import random
random.seed(11)
np.random.seed(17)

dataset, labels = pickle.load(open("../data/part2_dataset2.data", "rb"))

svm_parameter_grid = {"SVM__C": [0.1, 0.5],
              "SVM__kernel": ["poly", "rbf"]
}

# the cross validation
cross_validation = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=np.random.randint(1, 1000))

# Normalizing the data with the standardscaler
pipeline = Pipeline([('scaler', StandardScaler()), ('SVM', SVC())])


svmgrid = GridSearchCV(pipeline, svm_parameter_grid, scoring="accuracy", cv=cross_validation, verbose=True)
svmgrid.fit(dataset, labels)
#----------------------#
# Displayingg results #
print("The configurations are as follows: ")
print(svmgrid.cv_results_["params"], sep="\n")
print("-------------------------------------------------------\n")
for i in range(0,50):
    name = "split"+str(i)+"_test_score"
    splitResults = list(svmgrid.cv_results_[name])
    print("Split {} --> {}".format(i,splitResults))
    print("---------")

#print(svmgrid.cv_results_)
#print(svmgrid.cv_results_["params"])
means = svmgrid.cv_results_["mean_test_score"]
std = svmgrid.cv_results_["std_test_score"]
print("Mean test scores: ", svmgrid.cv_results_["mean_test_score"])
print("Std of test scores: ", svmgrid.cv_results_["std_test_score"])
print("Best mean score: ", svmgrid.best_score_)
print("Best parameters: ", svmgrid.best_params_)
print("Best Estimator: ", svmgrid.best_estimator_)

for i in range(4):
    confidence_interval_min = means[i] - (1.96 * std[i] / m.sqrt(50))  # finding min in confidence_interval
    confidence_interval_max = means[i] + (1.96 * std[i] / m.sqrt(50))  # finding max in confidence_interval
    print("Model {} PERFORMANCE Confidence Interval: {}".format(i,(confidence_interval_min,confidence_interval_max)))
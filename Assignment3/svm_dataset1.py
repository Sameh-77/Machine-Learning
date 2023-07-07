#======================================================#
#== Sameh Algharabli - CNG 409 -- Assignment3 --  ==#
#======================================================#
import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import seaborn as sns
from mlxtend.plotting import plot_decision_regions

dataset, labels = pickle.load(open("../data/part2_dataset1.data", "rb"))
svm_parameter_grid = {"C": [0.1, 0.5],
              "kernel": ["poly", "rbf"]
}

# This function is for plotting the boundaries #
def plotBoundaries(model,dataset,labels,title):

    # create a mesh to plot in
    x_min, x_max = dataset[:, 0].min() - 1, dataset[:, 0].max() + 1
    y_min, y_max = dataset[:, 1].min() - 1, dataset[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # getting the predictions #
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8,cmap =plt.cm.coolwarm)

    # Plotting the training points
    plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap =plt.cm.coolwarm)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

    plt.show()

# Nested loop for the parameters #
for c in svm_parameter_grid["C"]:
    for k in svm_parameter_grid["kernel"]:
        svm = SVC(C=c, kernel=k)
        svm.fit(dataset, labels)
        title = "C = " + str(c) + ", Kernel = " + k
        plotBoundaries(svm,dataset,labels,title)
        #plot_decision_regions(dataset,labels,clf = svm, legend =2)
        #plt.show()


#======================================================#
#== Sameh Algharabli - CNG 409 -- Assignment2 -- Kmeans Experiment ==#
#======================================================#
#--- Libraries ---#

from KMeans import KMeans
import pickle
import matplotlib.pyplot as plt
#-----------------------------------------------------#
# Loading the datasets #
dataset1 = pickle.load(open("../data/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../data/part2_dataset_2.data", "rb"))

#----------------------------------------------------#
# Nested loops to do the grid seach over the hyperparameters
# the k values from 2 to 10, and the datasets 1 or 2

k_values = [x for x in range(2, 11)]
datasets = [dataset1, dataset2]
i = 1
for dataset in datasets:
    results = []
    print("Running For dataset: ",i)
    print("---------------------------")
    # A for loop to go throguh all the k values
    for k in k_values:
        print("K = ",k)
        print("Repeating 10 times")
        # Repeat 10 times for each k
        k_loss = []
        for x in range(10):
            # creating the Kmeans object
            kmeansClassifier = KMeans(dataset, k)
            # getting the loss
            clustersCenters, clustersData, loss = kmeansClassifier.run()
            print("Repating: ",x)
            print("Cluster Centers: ",clustersCenters)
            print("Loss: ",loss)
            # Storing the losses for each k in k_loss list
            k_loss.append(loss)
        #when the 10 repeats are finished, I take the min loss to be the loss for k
        min_loss = min(k_loss)
        k_results = {"k": k, "min loss": min_loss}
        print("---------------------")
        print("Finished 10 repeats for k = ",k)
        print("Mean loss for k = {} is {}".format(k,min_loss))
        print("---------------------")
        results.append(k_results)
    print("Plotting k versus loss for dataset",i)
    scores = []
    for item in results:
        scores.append(item['min loss'])
    plt.plot(range(2, 11), scores)
    plt.xlabel('k')
    plt.ylabel('loss')
    title = "The Elbow for Kmeans on Dataset " + str(i)
    plt.title(title)
    plt.show()
    print("Completed for Dataset ",i)
    print("==============================================================")
    i += 1

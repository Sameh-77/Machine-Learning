#======================================================#
#== Sameh Algharabli - CNG 409 -- Assignment2 -- Kmeans ++==#
#======================================================#
#--- Libraries ---#

import numpy as np
from Distance import Distance
import operator
#-------------------------------------------------------#

class KMeansPlusPlus:
    def __init__(self, dataset, K=2):
        """
        :param dataset: 2D numpy array, the whole dataset to be clustered
        :param K: integer, the number of clusters to form
        """
        self.K = K
        self.dataset = dataset
        # each cluster is represented with an integer index
        # self.clusters stores the data points of each cluster in a dictionary
        self.clusters = {i: [] for i in range(K)}
        # self.cluster_centers stores the cluster mean vectors for each cluster in a dictionary
        self.cluster_centers = {i: None for i in range(K)}
        # you are free to add further variables and functions to the class

    # -------------------------------------------------------------------------#
    # This function calculates the loss after all the iterations are done
    def calculateLoss(self):
        """Loss function implementation of Equation 1"""

        error = 0
        for instance in self.dataset:
            for k in range(self.K):
                # If the instance exist in the cluster, calculate the loss
                if list(instance) in self.clusters[k]:
                    error += np.square(np.sum((instance - self.cluster_centers[k]) ** 2))
                # else, the loss will be zero
                else:
                    error = error + 0
        return error

    # -------------------------------------------------------------------#
    # This function is responsible for getting initial random initial centroids depending on the number of k

    def get_initial_centroids(self):

        data = self.dataset
        k = self.K


        # Assigning random centers from the dataset to the first clusters
        self.cluster_centers[0] = list(data[np.random.randint(data.shape[0]), :])

        ## compute remaining k - 1 centroids
        # Cluster 0 has been assigned, and now I'm assigning centers for the remaining centroids
        for c_id in range(1, k):

            dist = []
            # A for loop to go through all the instances in the data
            for instance in data:
                distances = []
                # a for loop to go through all the clusters centers
                for center in self.cluster_centers:
                    # If the center is < c_id, then calculate the distance, else, break and take another instance
                    if center >= c_id:
                        break
                    # calculating the distance between the instance and the cluster centers
                    distance = Distance.calculateMinkowskiDistance(instance, self.cluster_centers[center], 2)
                    distances.append([distance,instance])
                # sorting the distance to take the closest center to the points
                distances = sorted(distances, key=operator.itemgetter(0), reverse=False)
                # Appedning the distance and the distance and instance that gave min distance to the list that have
                # all the distance from all instances to that center
                dist.append([distances[0][0], distances[0][1]])

            # sorting the list of all distances to get the max distance
            dist = sorted(dist, key=operator.itemgetter(0), reverse=True)
            # getting the value of the instance that gave the maximum distance
            max_distance_instance = dist[0][1]
            # assigning that value to be the next cluster_center
            self.cluster_centers[c_id] = list(max_distance_instance)

    # -------------------------------------------------------------------#
    """This function is used to check if an instance exists in a specefic cluster
    It takes the instance and the closest centroid, check if it already belongs to the closes centroid, return -1
    which means "do not change anything", elif it belongs to a cluster and that cluster is not the closest, then 
    it returns the cluster number so that It will be remove from the cluster data, and appended to the data 
    of the closest centroid, else (if it does not belong to any cluster), return -2, so just append it to the closest
    cenntroid
    """
    def check_instance(self, instance, closest_centroid):

        for cluster in self.clusters:
            # If it belongs to a cluster
            if list(instance) in (self.clusters[cluster]):
                # if that cluster is the closestcentroid
                if cluster == closest_centroid:
                    return -1
                # if that cluster is not the closest centroid
                else:
                    return cluster
        # if it does not belong to any cluster
        return -2

    def run(self):
        """Kmeans algorithm implementation"""
        # Below is an explanation of how this function work
        # initialize centroids depending on the probability
        # while loop until old centroid_centers  = current centroid_centers
        # for each instance in the dataset, calculate the distance with all the centroids
        # take the index of the centroid that gave min distance
        # Check if the instance already exists in the clusters
        # assign that data to that cluster[index]
        # update the cluster_centers with the new mean of the data of that cluster
        # calculate loss

        # ----- Initialzie random centroids ------#
        self.get_initial_centroids()
        prev_centroids = {}

        # A while loop till prev_centroids = current centroids
        while (self.cluster_centers != prev_centroids):
            # For each instance in the dataset
            for instance in self.dataset:
                distances = []
                # for each center in the cluster centers
                for center in self.cluster_centers:
                    # calculate Eucl. distance between the instance and the center
                    distance = Distance.calculateMinkowskiDistance(instance, self.cluster_centers[center], 2)
                    distances.append([distance, center])
                # Sorting the distances to get the min centers
                distances = sorted(distances, key=operator.itemgetter(0), reverse=False)
                # getting the index of the center that gave min distance
                closest_centroid = distances[0][1]
                # check if the instance already exists in the clusters
                belongs_to_cluster = self.check_instance(instance, closest_centroid)
                # If the instance belongs to the data of the closest centroid, then continue
                if belongs_to_cluster == -1:
                    continue
                # if the data does not belong to any cluster
                elif belongs_to_cluster == -2:
                    self.clusters[closest_centroid].append(list(instance))
                # if the data belongs a cluster that is not the closest centroid
                else:
                    # remove the instance from the current cluster data
                    self.clusters[belongs_to_cluster].remove(list(instance))
                    # add the instance to the data of the closest centroid
                    self.clusters[closest_centroid].append(list(instance))

            # taking a copy of the current centroid to be previous centroids
            prev_centroids = self.cluster_centers.copy()

            # Updating cluster with the mean of the data #
            for cluster in self.clusters:
                # getting the data of each cluster
                cluster_data = self.clusters[cluster]
                # if there is no data, do not do anything, else update the cluster_centers
                if (len(cluster_data) != 0):
                    # calculating the mean of the data
                    data_mean = np.mean(cluster_data, axis=0)
                    # updating the cluster_centers to be the data mean
                    self.cluster_centers[cluster] = list(data_mean)

        # After the algorithms converges, I calculate the error or loss
        error = self.calculateLoss()

        return self.cluster_centers, self.clusters, error

#====================================================================================#

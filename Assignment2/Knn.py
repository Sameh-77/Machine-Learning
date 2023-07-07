#======================================================#
#== Sameh Algharabli - CNG 409 -- Assignment2 -- KNN ==#
#======================================================#
#--- Libraries ---#
from Distance import Distance
import operator
import numpy as np
#------------------------------------------------------#

class KNN:
    def __init__(self, dataset, data_label, similarity_function, similarity_function_parameters, k):
        """
        :param dataset: dataset on which KNN is executed, 2D numpy array
        :param data_label: class labels for each data sample, 1D numpy array
        :param similarity_function: similarity/distance function, Python function
        :param similarity_function_parameters: auxiliary parameter or parameter array for distance metrics
        :param K: how many neighbors to consider, integer
        """
        self.K = k
        self.dataset = dataset
        self.dataset_label = data_label
        self.similarity_function = similarity_function
        self.similarity_function_parameters = similarity_function_parameters
    #------------------------------------------------------------------------------#
    def predict(self, testdata):
        # this list will store the predictions for the test data
        predictions = []
        # A for loop to go through all the instances in the test data
        for x in range(0, len(testdata)):
            distances = []
            distance = 0
            # A for loop to go throguh all the instances in the train data
            for j in range(0, len(self.dataset)):
                # Checking the similarity function, according to that, calculate the distance
                if self.similarity_function == "cosine":
                    distance = Distance.calculateCosineDistance(self.dataset[j],testdata[x])
                elif self.similarity_function == "minkowski":
                    distance = Distance.calculateMinkowskiDistance(self.dataset[j],testdata[x], p=2)
                elif self.similarity_function == "mahalanobis":
                    V = np.cov(np.array([testdata[x], self.dataset[j]]).T)
                    IV = np.linalg.pinv(V)
                    distance = Distance.calculateMahalanobisDistance(self.dataset[j],testdata[x], IV)

                # Appending the resulting distance with the index of instance of the train data to distances list
                distances.append([self.dataset[j],distance,j])

            # Sorting the distances list according to the distance
            distances = sorted(distances,key=operator.itemgetter(1),reverse=False)
            neighbors = []
            # Now the distances are sorted, so create the neighbors depedning on the number of k
            for z in range(self.K):
                # Here i'm adding the distance and the index of the instance in the train data that gave that distance
                neighbors.append([distances[z][0],distances[z][2]])

            # Now we have the neighbors and the indices of them
            # using the indeces I can find the neighbours labels from the train labels
            classVotes = {}
            # A loop to go through all the neighbours
            for t in range(len(neighbors)):
                # getting the index of the neighbor
                response_index = neighbors[t][1]
                # getting the label of the neighbour from the train labels
                response = self.dataset_label[response_index]
                # If the label is there, increment its cound, else, add it and give it count 1
                if response in classVotes:
                    classVotes[response] += 1
                else:
                    classVotes[response] = 1

            # Sorting the votes to get the majority vote
            sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
            # Appending the vote that has highest count to the predictions
            predictions.append(sortedVotes[0][0])
        return predictions
    #-------------------------------------------------------------------------------------#
    """
    I added this function where I can evaluate the performance of my KNN 
    This function takes the predictions and the original test labels, and it return the accuracy  
    """
    def evaluate(self, current_test_labels, predicted):
        correct = 0
        for i in range(len(current_test_labels)):
                # if the prediction = original test labels, increment the correct count
                if current_test_labels[i] == predicted[i]:
                    correct += 1
        accuracy = correct/float(len(current_test_labels))
        return accuracy
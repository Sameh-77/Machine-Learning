#======================================================#
#== Sameh Algharabli - CNG 409 -- Assignment3 --  ==#
#======================================================#
import math
import numpy as np
import pandas as pd

# In the decision tree, non-leaf nodes are going to be represented via TreeNode
class TreeNode:
    def __init__(self, attribute):
        self.attribute = attribute
        # dictionary, k: subtree, key (k) an attribute value, value is either TreeNode or TreeLeafNode
        self.subtrees = {}


# In the decision tree, leaf nodes are going to be represented via TreeLeafNode
class TreeLeafNode:
    def __init__(self, data, label):
        self.data = data
        self.labels = label


class DecisionTree:
    def __init__(self, dataset, labels, features, criterion="information gain"):
        """
        :param dataset: array of data instances, each data instance is represented via an Python array
        :param labels: array of the labels of the data instances
        :param features: the array that stores the name of each feature dimension
        :param criterion: depending on which criterion ("information gain" or "gain ratio") the splits are to be performed
        """
        self.dataset = dataset
        self.labels = labels
        self.features = features
        self.criterion = criterion
        # it keeps the root node of the decision tree
        self.root = None
        self.entropy = 0
        # further variables and functions can be added...

    def calculate_entropy__(self, dataset, labels):
        """
        :param dataset: array of the data instances
        :param labels: array of the labels of the data instances
        :return: calculated entropy value for the given dataset
        """
        entropy_value = 0.0

        """
        Entropy calculations
        """
        pos = 0
        neg = 0
        # Taking each label in the labels, count how many ones and how many zeros, and calculate the entropy accordingly
        for label in labels:
            if label == 1:
                pos += 1
            else:
                neg += 1
        if pos == 0 or neg == 0:
            return 0
        else:
            p = pos / (pos + neg)
            n = neg / (pos + neg)
            entropy_value = -(p * math.log(p, 2) + n * math.log(n, 2))

        return entropy_value

    def calculate_average_entropy__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an average entropy value is calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute an average entropy value is going to be calculated...
        :return: the calculated average entropy value for the given attribute
        """
        average_entropy = 0.0
        """
            Average entropy calculations
        """

        uniq = np.unique(dataset[attribute])
        # calculating the average entropy for each feature by calculating the entropies for each value in the feature
        for val in uniq:
            subdata = dataset[dataset[attribute] == val]
            sublabels = subdata["Play Golf"]
            average_entropy += self.calculate_entropy__(subdata, sublabels) * (subdata.shape[0]/ dataset.shape[0])
        #print("Average entropy: ",average_entropy)
        return average_entropy

    def calculate_information_gain__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an information gain score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the information gain score is going to be calculated...
        :return: the calculated information gain score
        """
        information_gain = 0.0
        """
            Information gain calculations
        """
        # infomration gain for a feature = total entropy - averageEntropy for that feature
        self.entropy = self.calculate_entropy__(dataset, labels)
        information_gain = self.entropy - self.calculate_average_entropy__(dataset,labels,attribute)
        #print("gain = ",information_gain)
        return information_gain

    def calculate_intrinsic_information__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances on which an intrinsic information score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the intrinsic information score is going to be calculated...
        :return: the calculated intrinsic information score
        """
        intrinsic_info = 0.0
        """
            Intrinsic information calculations for a given attribute
        """
        uniq = np.unique(dataset[attribute])
        len_total = dataset.shape[0]
        for val in uniq:
            subdata = dataset[dataset[attribute] == val]
            len_subdata = subdata.shape[0]
            z = len_subdata/len_total
            intrinsic_info += z * math.log(z, 2)

        return (intrinsic_info*-1)

    def calculate_gain_ratio__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances with which a gain ratio is going to be calculated
        :param labels: array of labels of those instances
        :param attribute: for which attribute the gain ratio score is going to be calculated...
        :return: the calculated gain ratio score
        """
        """
            Your implementation
        """
        information_gain = self.calculate_information_gain__(dataset,labels,attribute)
        intrinsic_info = self.calculate_intrinsic_information__(dataset,labels, attribute)

        gain_ratio = information_gain/intrinsic_info

        return gain_ratio


    def ID3__(self, dataset, labels, used_attributes):
        """
        Recursive function for ID3 algorithm
        :param dataset: data instances falling under the current  tree node
        :param labels: labels of those instances
        :param used_attributes: while recursively constructing the tree, already used labels should be stored in used_attributes
        :return: it returns a created non-leaf node or a created leaf node
        """
        """
            Your implementation
        """

        max_gain = 0.0
        max_feat = ""
        # print("data = \n", dataset)
        # print("--------")
        # print("labels = \n", labels)
        # print("--------")

        # If the entropy is 0, it means it's a leaf node, so return the leaf node
        if self.calculate_entropy__(dataset, labels) ==0:
            leaf = TreeLeafNode(dataset, labels)
            return leaf

        # checking the used attributes and the not used attributes
        if len(used_attributes) == 0:
            features = self.features
        else:
            set1 = set(self.features)
            set2 = set(used_attributes)
            features = list(set1 - set2) # getting the list of the unused attributes

        # A loop for each feature
        for feature in features:
            # checking the criterion
            if self.criterion == "information gain":
                gain = self.calculate_information_gain__(dataset,labels,feature)
            else:
                gain = self.calculate_gain_ratio__(dataset, labels, feature)
            #print("Feature : {}, gain: {}".format(feature,gain))
            # getting the feature that got best gain
            if gain >= max_gain:
                max_gain = gain
                max_feat = feature

        # Creating a tree node with max_feature as attribute
        node = TreeNode(max_feat)
        used_attributes.append(max_feat)
        uniq = np.unique(dataset[max_feat])
        # A loop to go through all the values of the max feature
        for u in uniq:
            # getting only the data where the max feature value = specefic value
            # e.g. getting all the data where outlook = sunny
            subdata = dataset[dataset[max_feat] == u]
            sublabels = subdata["Play Golf"]
            # recursion to continue building the tree
            node.subtrees[u] = self.ID3__(subdata, sublabels, used_attributes)
        # returning the root node
        return node




    def predict(self, x):
        """
        :param x: a data instance, 1 dimensional Python array
        :return: predicted label of x

        If a leaf node contains multiple labels in it, the majority label should be returned as the predicted label
        """
        predicted_label = None
        """
            Your implementation
        """
        roott = self.root
        attribute_test = None
        # A loop until i reach a leaf node
        while not isinstance(roott, TreeLeafNode):
            # getting the attribute of the root
            attribute = roott.attribute
            #print("attribute is: ",attribute)
            # checking the index of the attribute in the dataset
            # and getting the value of that attribute in the test sample
            if(attribute == "Temperature"):
                attribute_test = x[0]
            elif(attribute == "Outlook"):
                attribute_test = x[1]
            elif (attribute == "Humidity"):
                attribute_test = x[2]
            elif (attribute == "Windy"):
                attribute_test = x[3]
            # going to the subtrees where the key is the value of the attribute in the test sample
            # updating the root with its subtrees
            roott = roott.subtrees[attribute_test]
        # getting the predicted label, since each leaf node has more than one label, and they are all the same
        # Im just taking the first one
        predicted_label = list(roott.labels)[0]

        return predicted_label

    def train(self):
        # Here, before I call the ID3, i create a dataframe that contains, features, dataset and labels, just
        # to make the implementation  easier
        for i in range(len(self.dataset)):
            self.dataset[i].append(self.labels[i])

        features2 = [item for item in self.features]
        features2.append("Play Golf")
        # print("features: ",features)
        # print("features2: ",features2)
        self.dataset = pd.DataFrame(self.dataset, columns=features2)
        self.root = self.ID3__(self.dataset, self.labels, [])
        print("Training completed")
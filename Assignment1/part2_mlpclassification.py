#---------------------------------------------------------------------------------------#
# Sameh Algharabli -- 2591386 -- part 2 mlpclassification # 
#---------------------------------------------------------------------------------------#

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

# this function returns the neural network output for a given dataset and set of parameters
def forward_pass(w1, b1, w2, b2, input_data):
    """
    The network consists of 3 inputs, 16 hidden units, and 3 output units
    The activation function of the hidden layer is sigmoid.
    The output layer should apply the softmax function to obtain posterior probability distribution. And the function should return this distribution
    Here you are expected to perform all the required operations for a forward pass over the network with the given dataset
    """
    hidden_layer = torch.matmul(input_data,w1) + b1  # sum of products of the input layer 
    hidden_layer_output = torch.sigmoid(hidden_layer) # calling the sigmoid function (activation function)to find the units in the hidden layer 

    output_layer = torch.matmul(hidden_layer_output,w2) + b2 # sum of products of the hidden layer units and the weights from the hidden layer 
    output = torch.softmax(output_layer,dim=1)  # applying the softmax function to the output layer 
    return output

# we load all training, validation, and test datasets for the classification task
train_dataset, train_label = pickle.load(open("data/part2_classification_train.data", "rb"))
validation_dataset, validation_label = pickle.load(open("data/part2_classification_validation.data", "rb"))
test_dataset, test_label = pickle.load(open("data/part2_classification_test.data", "rb"))

# when you inspect the training dataset, you are going to see that the class instances are sequential (e.g [1,1,1,1 ... 2,2,2,2,2 ... 3,3,3,3])
# we shuffle the training dataset by preserving instance-label relationship
indices = list(range(len(train_dataset)))
np.random.shuffle(indices)
train_dataset = np.array([train_dataset[i] for i in indices], dtype=np.float32)
train_label = np.array([train_label[i] for i in indices], dtype=np.float32)

# In order to be able to work with Pytorch, all datasets (and labels/ground truth) should be converted into a tensor
# since the datasets are already available as numpy arrays, we simply create tensors from them via torch.from_numpy()
train_dataset = torch.from_numpy(train_dataset)
train_label = torch.from_numpy(train_label)

validation_dataset = torch.from_numpy(validation_dataset)
validation_label = torch.from_numpy(validation_label)

test_dataset = torch.from_numpy(test_dataset)
test_label = torch.from_numpy(test_label)


# w1 defines the parameters between the input layer and the hidden layer, initialize w1 via the Normal distribution (mean=0, std=1).
w1 = torch.from_numpy(np.random.normal(0,1,size=(3,16)).astype(np.float32)).requires_grad_(True)

# b defines the bias parameters for the hidden layer, initialize b1 via the Normal distribution (mean=0, std=1).
b1 = torch.from_numpy(np.random.normal(0,1,size=(1,16)).astype(np.float32)).requires_grad_(True)

# w2 defines the parameters between the hidden layer and the output layer,initialize w2 via the Normal distribution (mean=0, std=1).
w2 = torch.from_numpy(np.random.normal(0,1,size=(16,3)).astype(np.float32)).requires_grad_(True)

# and finally, b2 defines the bias parameters for the output layer, initialize b2 via the Normal distribution (mean=0, std=1).
b2 = torch.from_numpy(np.random.normal(0,1,size=(1,3)).astype(np.float32)).requires_grad_(True)


# This function calculates the cross entropy loss 
def cross_entropy(groundTruth,prediction):  
    loss = -torch.sum(groundTruth * torch.log(prediction)) 
    return loss/float(prediction.shape[0]) # finding the mean 
    

# This function calculates the accuracy     
def calc_accuracy(groundTruth,prediction,no_of_samples):
    _,out = torch.max(prediction,1) # getting the index of the highest value 
    _,actualIndex = torch.max(groundTruth,1)  # getting the index of the class ( which is the highest) 
    i=0
    correct =0
    # checking if the index of the prediction is similar to the index of the class ,then prediction is correct 
    for i in range(len(actualIndex)): 
        if (actualIndex[i] == out[i]):
            correct +=1
    
    correct = (100*correct/no_of_samples)
    return correct

# you are expected to use the stochastic gradient descent optimizer
# w1, b1, w2 and b2 are the trainable parameters of the neural network
optimizer = torch.optim.SGD([w1, b1, w2, b2], lr=0.001)

# These arrays will store the loss values incurred at every training iteration
iteration_array = []
train_loss_array = []
validation_loss_array = []

# We are going to perform the backpropagation algorithm 'ITERATION' times over the training dataset
# After each pass, we are calculating the average/mean cross entropy loss over the validation dataset along with accuracy scores on both datasets.
ITERATION = 15000
for iteration in range(1, ITERATION+1):
    iteration_array.append(iteration)

    
    # we need to zero all the stored gradient values calculated from the previous backpropagation step.
    optimizer.zero_grad()
    # Using the forward_pass function, we are performing a forward pass over the network with the training data
    train_predictions = forward_pass(w1, b1, w2, b2, train_dataset)
    # calculate the MEAN cross-entropy loss with respect to the network predictions and the training label
    train_mean_crossentropy_loss = cross_entropy(train_label,train_predictions)

    
    train_loss_array.append(train_mean_crossentropy_loss.item())

    # We initiate the gradient calculation procedure to get gradient values with respect to the calculated loss 
    train_mean_crossentropy_loss.backward()
    # After the gradient calculation, we update the neural network parameters with the calculated gradients.
    optimizer.step()

    # after each epoch on the training data we are calculating the loss and accuracy scores on the validation dataset
    # with torch.no_grad() disables gradient operations, since during testing the validation dataset, we don't need to perform any gradient operations
    with torch.no_grad():
        # calculate the accuracy score on the training dataset by using the training labels.
        train_accuracy = calc_accuracy(train_label,train_predictions,train_label.shape[0])

        validation_predictions = forward_pass(w1, b1, w2, b2, validation_dataset)
        
        # calculate the average/mean cross entropy loss for the validation datasets by using the validation dataset labels.
        validation_mean_crossentropy_loss = cross_entropy(validation_label,validation_predictions)#(-torch.sum(validation_label * torch.log(validation_predictions)).mean())

        validation_loss_array.append(validation_mean_crossentropy_loss.item())

        # calculate the accuracy score on the validation dataset
        validation_accuracy = calc_accuracy(validation_label,validation_predictions,validation_label.shape[0])
    
    print("Iteration : %d - Train Loss %.4f - Train Accuracy : %.2f - Validation Loss : %.4f Validation Accuracy : %.2f" % (iteration+1, train_mean_crossentropy_loss.item(), train_accuracy, validation_mean_crossentropy_loss.item(), validation_accuracy))


# after completing the training, we calculate our network's accuracy score on the test dataset...
# Again, here we don't need to perform any gradient-related operations, so we are using torch.no_grad() function.
with torch.no_grad():
    test_predictions = forward_pass(w1, b1, w2, b2, test_dataset)
    # calculate the network accuracy score on the test dataset...
    test_accuracy = calc_accuracy(test_label,test_predictions,test_label.shape[0])
    print("Test accuracy : %.2f" % (test_accuracy))
    

# We plot the loss versus iteration graph for both datasets (training and validation)
plt.plot(iteration_array, train_loss_array, label="Train Loss")
plt.plot(iteration_array, validation_loss_array, label="Validation Loss")
plt.legend()
plt.show()






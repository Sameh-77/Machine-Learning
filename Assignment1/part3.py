#---------------------------------------------------------------------------------------#
# Sameh Algharabli -- 2591386 -- part 3#  hyperparameter search for the MLP algorithm/method by utilizing a validation dataset
#---------------------------------------------------------------------------------------#


import torch
import torch.nn as nn
import numpy as np
import pickle
import math as m

class MLPModel(nn.Module):
    # getting the parameters when creating the model # 
    def __init__(self,layers,neurons,activation):
        super(MLPModel,self).__init__()
        
        # I'm doing 1 or 2 hidden layers only, so depending on that I'm creating my layers 
        self.layers=layers
        if (layers==1):
            self.layer1 = nn.Linear(784, neurons)
            self.layer2 = nn.Linear(neurons,10) 
        else: 
            # for the number of neuron for the second hidden layer, I just make it 784 - neuorons sent 
            self.layer1 = nn.Linear(784, neurons)
            self.layer2 = nn.Linear(neurons,784-neurons)
            self.layer3 = nn.Linear(784-neurons,10)
        
        self.activation_function= activation

    
    
    def forward(self,x):
        # depending on the number of layers, I'm doing the activation and the outputs 
        hidden_layer1_output = self.activation_function(self.layer1(x))
        if self.layers == 1: 
            output_layer = self.layer2(hidden_layer1_output)
        else:
            hidden_layer2_output=  self.activation_function(self.layer2(hidden_layer1_output))
            output_layer = self.layer3(hidden_layer2_output)  
        return output_layer

#-----------------------------------------------------------------#


# we load all the datasets of Part 3
x_train, y_train = pickle.load(open("data/mnist_train.data", "rb"))
x_validation, y_validation = pickle.load(open("data/mnist_validation.data", "rb"))
x_test, y_test = pickle.load(open("data/mnist_test.data", "rb"))

x_train = x_train/255.0
x_train = x_train.astype(np.float32)

x_test = x_test / 255.0
x_test = x_test.astype(np.float32)

x_validation = x_validation/255.0
x_validation = x_validation.astype(np.float32)

# and converting them into Pytorch tensors in order to be able to work with Pytorch
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).to(torch.long)

x_validation = torch.from_numpy(x_validation)
y_validation = torch.from_numpy(y_validation).to(torch.long)

x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test).to(torch.long)
#-------------------------------------------------#


loss_function = nn.CrossEntropyLoss()

""" Here I have the sets which have my different hyperparameters
I will do nested loops to go through each set"""

no_of_hidden_layers = {1,2}
no_of_neuorons = {64,256} #,128,32} 
learning_rate = {0.001,0.0001}
number_of_iterations = {70,200}  #,150}
activation_functions = {'sigmoid' : nn.Sigmoid(), 'LeakyReLU': nn.LeakyReLU()} ##'reLU': nn.ReLU(),'Tanh': nn.Tanh() ,}

#-------------------------------------#

softmax_function = torch.nn.Softmax(dim=1)

#------------------------------------#

#This function prints the parameters of each model# 
def print_model(model):
    print("****************************************")
    print("********   Model Specifications  *******\n")
    #print("Model Repeat: " + str(repeat))
    for key,value in model.items():
        print(key + ": " + str(value)) 
    print("*************************************\n")
#------------------------------------#     

# This function calculates the accuracy   
def calc_accuracy(groundTruth,prediction,no_of_samples):
    # print(prediction.shape)
    # print(groundTruth.shape)
    # print(prediction)
    # print(groundTruth)
    _,out = torch.max(prediction,1) # finding the index of the max value (it gives the class number) 
    i=0
    correct =0
    for i in range(len(groundTruth)): 
        if (groundTruth[i] == out[i]):
            correct +=1
    
    correct = (100*correct/no_of_samples)
    return correct
#------------------------------------#

   
""" function is used to train the model 
I send the parameters to it with the Xtrain and Ytrain sets
flag = 0 means I'm not done with tuning, so train with old sets and return validation 
flag = 1 means I'm done with tuning, so train with new sets and return the model 
"""
def trainModel(num_of_layers,num_of_neorons,activation,learning_rate,iteration,Xtrain,Ytrain,flag):
    nn_model = MLPModel(num_of_layers,num_of_neorons,activation)
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate)
    ITERATION = iteration 
    for iteration in range(1, ITERATION +1):
        # we need to zero all the stored gradient values calculated from the previous backpropagation step.
        optimizer.zero_grad()
        predictions = nn_model(Xtrain)
        
        loss_value = loss_function(predictions,Ytrain)
        
        # We initiate the gradient calculation procedure to get gradient values with respect to the calculated loss 
        loss_value.backward()
        
        # After the gradient calculation, we update the neural network parameters with the calculated gradients
        optimizer.step()
        
        # after each epoch on the training data we are calculating the loss and accuracy scores on the validation dataset
        # with torch.no_grad() disables gradient operations, since during testing the validation dataset, we don't need to perform any gradient operations
        with torch.no_grad():
            train_prediction = nn_model(Xtrain)
            train_loss = loss_function(train_prediction,Ytrain)
            train_accuracy = calc_accuracy(Ytrain,train_prediction,Ytrain.shape[0])
            if flag==0:
                predictions = nn_model(x_validation)
                probability_score_valies = softmax_function(predictions) 
                validation_loss = loss_function(predictions,y_validation)
                validation_accuracy = calc_accuracy(y_validation,predictions,y_validation.shape[0])
                  
        if flag==0:
            print("Iteration: %d Training Loss: %f - Training Accuracy= %.2f- Validation Loss = %f - Validation Accuracy = %.2f" %(iteration,train_loss.item(),train_accuracy,validation_loss.item(),validation_accuracy))
        else:
            print("Iteration: %d Training Loss: %f - Training Accuracy= %.2f" %(iteration,train_loss.item(),train_accuracy))          
    if flag==0:
        return nn_model,validation_accuracy
    else:
        return nn_model
    

model = {}
models_list = [] 
model_counter = 0 

"""Nested loops to go through all of the model sets """ 
# Iterate through the learningrate set
for item in learning_rate:      
    # Iterate through the no_of_hidden_layers set
    for layers in no_of_hidden_layers:
        # Iterate through the no_of_neuorons set 
        for neuron in no_of_neuorons:
            # Iterate through the activation_functions sets
            for activation in activation_functions.keys():
                # Iterate through the number_of_iterations set 
                for iteration in number_of_iterations:
                    model_counter += 1 
                    same_model_validations = [] # this list is used to save the validation accuracies of the same model 
                    # creating same model with same parameters 10 times 
                    for repeat in range(10):
                        # saving the information of my model 
                        model = {"Model Number": model_counter, "Model Repeat":repeat+1, "Number of hidden layers":layers, "Number of neurons" : neuron, "Learning rate" : item, "Activation function" : activation_functions[activation] ,"Number of iterations": iteration, "Validation Accuracy": 0, "Confidence Interval": []}
                        activationFunc = activation_functions[activation]
                        print_model(model) # printing the models details 
                        
                        # creating the model and getting the validation_accuracy as an output 
                        _,validation_accuracy=trainModel(layers,neuron,activationFunc,item,iteration,x_train,y_train,0)
                        same_model_validations.append(validation_accuracy) # appending the validation_accuracy to the same_model_validations list 
                            
                        print("-------------------------------------\n\n")
                        
                    print(same_model_validations)  # print the validations of the same model after 10 repeats 
                    meann = sum(same_model_validations)/len(same_model_validations) # finding the mean of the validations 
                    standard_deviation = np.std(same_model_validations) # calculating standard_deviation
                    confidence_interval_min = meann - (1.96* standard_deviation/m.sqrt(10))  # finding min in confidence_interval
                    confidence_interval_max = meann + (1.96* standard_deviation/m.sqrt(10))  # finding max in confidence_interval
                    print("Mean= " + str(meann)) 
                    model["Validation Accuracy"] = meann # updating the validation_accuracy in the model 
                    model["Confidence Interval"] = [confidence_interval_min,confidence_interval_max] # updating the Confidence Interval in the model 
                    print("\n\nModel %d - Mean validation accuracy = %.2f - Confidence Interval: (%f,%f)\n" %(model["Model Number"],meann,confidence_interval_min,confidence_interval_max))
                    temp = model
                    models_list.append(temp) # appending the model the models list 
                    print("\n============CURRENT MODELS LIST===============\n")
                    for modell in models_list:
                        print_model(modell)
                    print("======================================================================")    
                    
#--------------------------------------------------------------------------------------------#
                    
"""Finding the best accuracy and printing the hyperparameters of the best model"""                    
print("*****************************")
maximum = 0
index = 1 

# printing all the models and their validation_accuracies 
accuracies_list = [] 
for model in models_list:
    accuracies_list.append(model["Validation Accuracy"])

print("-------------------")
print("Accuracies List") 
for item in accuracies_list: 
    print("Model Number: %d - Validation Accuracy= %f" %(index,item))
    index += 1                        
#---------------------------------------------#
    
# finding the max validation_accuracy    
print("------------------------------------")
maximum = max(accuracies_list) 
maximum_index = accuracies_list.index(maximum)
print("The model that gave maximum accuracy: \nModel Number: %d - Max Validation Accuracy= %f" %(maximum_index+1,maximum))

#print("The paramaeters of the model that gave maximum accuracy\n")
max_model = {} 
for item in models_list:
    if item['Model Number']== (maximum_index+1):    
        max_model= item 
        
        
print("****************************************")
print("********   BEST Model *******\n")
print_model(max_model)
    
#---------------------------------------------------------#

# appending both datasets " 

newXtrain = torch.cat((x_train,x_validation))
newYtrain = torch.cat((y_train,y_validation))

#-------------------------------#
# training the best model # 
                       
# print(max_model["Number of hidden layers"])
# print(max_model["Number of neurons"])
# print(max_model["Activation function"])
# print(max_model["Learning rate"])
# print(max_model["Number of iterations"])

print("*****************************")
print("Training The Best Model with the combined dataset")
print("-------------------------------------------------")
trainedModel = trainModel(max_model["Number of hidden layers"],max_model["Number of neurons"],
                    max_model["Activation function"],max_model["Learning rate"],max_model["Number of iterations"],newXtrain,newYtrain,1)

                    
with torch.no_grad():
    trainedModel.eval()
    predictions= trainedModel(x_test)
    test_loss = loss_function(predictions,y_test)
    test_accuracy = calc_accuracy(y_test,predictions,y_test.shape[0])
    print("Test Loss = %.2f" % (test_loss))
    print("Test Accuracy = %.2f" % (test_accuracy))

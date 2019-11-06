import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
train = pd.read_csv(sys.argv[1],header=None)
X_train = train.iloc[:,:-1].values
Y_train = train.iloc[:,-1].values
print(X_train.shape,Y_train.shape)
with open(sys.argv[2]) as f:
    param = f.read().split("\n")
    learningType = int(param[0])
    learningRate = float(param[1])
    maxIteration = int(param[2])
    batchSize = int(param[3])
    layer = list(map(int,param[4].split(" ")))
    layer.insert(0,X_train.shape[1])
    layer.append(1)
print(learningType,learningRate,maxIteration,batchSize,layer,X_train.shape)
def initialiseWeights(layer):
    weights = []
    bias = []
    for i in range(1,len(layer)):
        weights.append(np.zeros(layer[i-1]*layer[i]).reshape(layer[i],layer[i-1]))
        bias.append(np.zeros(layer[i]).reshape(layer[i],1))
        print(weights[i-1].shape,bias[i-1].shape)
    return (weights,bias)
def sigmoid(z):
    return 1/(1+np.exp(-z))
def feedforward(weights,bias,X_train):
    a_layer = []
    z_layer = []
    a_layer.append(X_train.T)
    z_layer.append(0)
    for i in range(0,len(layer)-1):
        z_layer.append(np.matmul(weights[i],a_layer[i])+bias[i])
        a_layer.append(sigmoid(z_layer[i+1]))
    return (a_layer,z_layer)
def backpropagate(weights,bias,a_layer,z_layer,X_train,Y_train):
    new_weights = []
    new_bias = []
    layer_index = len(a_layer)-1
    error_der = (a_layer[layer_index]-Y_train)/((1-a_layer[layer_index])*a_layer[layer_index]*X_train.shape[0])
    s = sigmoid(z_layer[layer_index])
    sigmaDer = (s*(1-s))
    delta = error_der*sigmaDer
    new_weights.append(np.subtract(weights[layer_index-1] ,np.multiply(learningRate,np.matmul(delta,a_layer[layer_index-1].T))))
    b = np.sum(delta,axis=1)
    new_bias.append(np.subtract(bias[layer_index-1],np.multiply(learningRate,b.reshape(b.shape[0],1))))
    layer_index-=1
    while(layer_index>0):
        s = sigmoid(z_layer[layer_index])
        sigmaDer = (s*(1-s))
        delta = np.matmul(weights[layer_index].T,delta)*sigmaDer
        new_weights.append(np.subtract(weights[layer_index-1] ,np.multiply(learningRate,np.matmul(delta,a_layer[layer_index-1].T))))
        b = np.sum(delta,axis=1)
        new_bias.append(np.subtract(bias[layer_index-1],np.multiply(learningRate,b.reshape(b.shape[0],1))))
        layer_index-=1
    new_bias.reverse()
    new_weights.reverse()
    return (new_weights,new_bias)
k = batchSize
l = X_train.shape[0]
weights, bias = initialiseWeights(layer)
error = []
batches = l/k
for i in range(maxIteration):
    start_index = int((k)*(i%batches))
    end_index = int((k)*((i%batches)+1))
    a_layer , z_layer = feedforward(weights,bias,X_train[start_index:end_index,:])
    weights, bias = backpropagate(weights,bias,a_layer,z_layer,X_train[start_index:end_index,:],Y_train[start_index:end_index])
a = np.array([])
for i in range(len(bias)):
    a = np.concatenate((a,bias[i].flatten(),weights[i].flatten('F')))
np.savetxt(sys.argv[3],a)
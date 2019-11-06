import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
import sys
from skimage import feature

np.random.seed(1234)
train = pd.read_csv(sys.argv[1],header=None)
test  = pd.read_csv(sys.argv[2],header=None)
classes = pd.get_dummies(train[1024],prefix="class_")
train_class = train.iloc[:,-1].values
train = pd.concat([train.iloc[:,:-1],classes],axis=1)
def initialiseWeights(layer):
    weights = []
    bias = []
    for i in range(1,len(layer)):
        weights.append((np.random.rand(layer[i-1]*layer[i]).reshape(layer[i],layer[i-1])-0.5))
        bias.append((np.random.rand(layer[i]).reshape(layer[i],1)-0.5))
    return (weights,bias)    
def normaliseFeatures(X_train):
    return (X_train-np.mean(X_train,axis=0))/255;
def gabor_filter(X_train,kernal):
    X_filtered = [ ndi.convolve(X_train[i],kernal) for i in range(X_train.shape[0])]
    return np.array(X_filtered)
def sigmoid(z):
    return 1/(1+np.exp((-1)*z))
def sigmoid_der(z):
    s = sigmoid(z)
    return s*(1-s)
def tanh(z):
    return (2/(1+np.exp((-2)*z)))-1
def tanhder(z):
    t = tanh(z)
    return 1-t*t;
def relu(z):
    return np.where(z>0,z,0.01*z)
def reluder(z):
    z[z>0]=1
    z[z<0]=0.01
    return z
def softmax(z):
    z = np.exp(z)
    return z/np.sum(z,axis=0)    
def hot_encode(Y_pred,Y_actual):
    Y_pred = Y_pred.T  
    Y_actual = Y_actual.T
    maxVal = np.max(Y_pred,axis=1).reshape(Y_pred.shape[0],1) 
    Y_pred = Y_pred-maxVal
    Y_pred[Y_pred==0]=1
    Y_pred[Y_pred<0]=0
    d = np.sum(Y_pred*Y_actual)
    return d/Y_actual.shape[0]    
def feedforward(weights,bias,X_train):
    a_layer = []
    z_layer = []
    a_layer.append(X_train.T)
    z_layer.append(0)
    for i in range(0,len(layer)-2):
        z_layer.append(np.matmul(weights[i],a_layer[i])+bias[i])
        # a_layer.append(sigmoid(z_layer[i+1]))
        a_layer.append(relu(z_layer[i+1]))
        # a_layer.append(tanh(z_layer[i+1]))
    z_layer.append(np.matmul(weights[-1],a_layer[-1])+bias[-1])
    a_layer.append(softmax(z_layer[-1]))    
    return (a_layer,z_layer)    
def backpropagate(weights,bias,a_layer,z_layer,X_train,Y_train,iteration,reg):
    regul = 1 - (reg*learningRate/Y_train.shape[0])
    new_weights = []
    new_bias = []
    layer_index = len(a_layer)-1
    delta = (a_layer[-1]-Y_train.T)/Y_train.shape[0]
    new_weights.append(np.subtract(regul*weights[layer_index-1] ,np.multiply((learningRate/iteration),np.matmul(delta,a_layer[layer_index-1].T))))
    b = np.sum(delta,axis=1)
    new_bias.append(np.subtract(regul*bias[layer_index-1],np.multiply((learningRate/iteration),b.reshape(b.shape[0],1))))
    layer_index-=1
    while(layer_index>0):
        # delta = np.matmul(weights[layer_index].T,delta)*sigmoid_der(z_layer[layer_index])
        delta = np.matmul(weights[layer_index].T,delta)*reluder(z_layer[layer_index])
        # delta = np.matmul(weights[layer_index].T,delta)*tanhder(z_layer[layer_index])
        new_weights.append(np.subtract(regul*weights[layer_index-1] ,np.multiply((learningRate/iteration),np.matmul(delta,a_layer[layer_index-1].T))))
        b = np.sum(delta,axis=1)
        new_bias.append(np.subtract(regul*bias[layer_index-1],np.multiply((learningRate/iteration),b.reshape(b.shape[0],1))))
        layer_index-=1
    new_bias.reverse()
    new_weights.reverse()
    return (new_weights,new_bias)  
maxaccuracy = 0
kernel = np.real(gabor_kernel(1, theta=np.pi/2,sigma_x=1, sigma_y=1))
X_train = normaliseFeatures(train.iloc[:,:-10].values)
Y_train = train.iloc[:,-10:].values
X_test  = normaliseFeatures(test.iloc[:,:-1].values)
# H = np.array([np.array(feature.hog(x, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1") for x in X_train)]
# print(H.shape)
# X_train = np.concatenate([X_train,gabor_filter(X_train,kernel[1])],axis=1)
# X_test = np.concatenate([X_test,gabor_filter(X_test,kernel[1])],axis=1)
learningType = 2
learningRate = float(0.3)
maxIteration = int(3000)
batchSize = int(80)
layer = [315]
layer.insert(0,X_train.shape[1])
layer.append(Y_train.shape[1])
print(learningType,learningRate,maxIteration,batchSize,layer)
print(X_train.shape,Y_train.shape,X_test.shape)

iteration = 0
k = batchSize
l = X_train.shape[0]
weights, bias = initialiseWeights(layer)
error_value = []
o = maxIteration
batches = l/k
for i in range(o):
    start_index = int(k*(i%batches))
    end_index = int(k*((i%batches)+1))
    print(i,end="\r",flush=True)
    a_layer , z_layer = feedforward(weights,bias,X_train[start_index:end_index,:])
    # y , z = feedforward(weights,bias,X_train)
    # ac = hot_encode(a_layer[-1],Y_train[start_index:end_index,:].T)
    # if(maxaccuracy<ac):
    #     iteration = i+1
    #     maxaccuracy = ac
    # error_value.append(error(a_layer[-1],Y_train[start_index:end_index,:]))
    weights, bias = backpropagate(weights,bias,a_layer,z_layer,X_train[start_index:end_index,:],Y_train[start_index:end_index,:],np.sqrt(1),0.21)        
# print(maxaccuracy,iteration)    
a_layer , z = feedforward(weights,bias,X_train)
print(hot_encode(a_layer[-1],Y_train.T))
a_layer , z_layer = feedforward(weights,bias,X_test)
a = np.array([ np.where(out==np.amax(out))[0][0] for out in a_layer[-1].T])
np.savetxt(sys.argv[3],a)    
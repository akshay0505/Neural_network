#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 20:14:25 2019

@author: kryntom
"""
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt

my_data = np.genfromtxt('data.csv', delimiter=',') # data reading
X1 = my_data[1:, 0].reshape(-1,1) # create simple X1 matrix
y = my_data[1:, 1].reshape(-1,1) # create y matrix
plt.scatter(X1, y) # plotting X1 gives idea of 3 degree equation
X2 = np.zeros([X1.shape[0], 1]) # array of zero for squres 
X3 = np.zeros([X1.shape[0], 1]) # array of zero for cubes
X4 = np.ones([X1.shape[0],1]) # array of ones for constant
X = np.zeros([X1.shape[0], 4]) # array of zero for final matrix
np.square(X1, out = X2) #creating squares of X matrix
np.power(X1, 3, out = X3)# creating cubes of X matrix

np.concatenate((X4,X1,X2,X3), out = X, axis = 1) # Making final matrix

# setting hyper parameters ( chosen so as to minimize MSE) 
#Note:- To minimize mse i have taken iters to be 1000000 but due to that the time of compilation is high
# to prevent this we can reduce it by a factor of 10 but that will result in less accuracy 
# Thus that will depend on the requirement of this code
alpha = 1
iters = 1000000
theta = np.zeros([1,4])


#Cost Computation
def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))


#gradient descent function
def gradientDescent(X,y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        print(i,end="\r",flush=True)
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = computeCost(X, y, theta)
    
    return theta,cost

#Finding the g values
g,cost = gradientDescent(X,y,theta,iters,alpha)
print(g)

# creating the final matrix(predicted values)
y_vals = g[0][0]*X4 + g[0][1]*X1 + g[0][2]*X2 + g[0][3]*X3

plt.scatter(X1, y_vals)

# finding the MSE
sum1 = 0  
n = len(y) 
for i in range (0,n):  
  dif= y[i] - y_vals[i]  
  sq_dif = dif**2 
  sum1= sum1+ sq_dif
MSE = sum1/n
print ("The Mean Square Error is: " , MSE)
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:13:56 2020

@author: 90538
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("CanserData.csv")

#print(data.info())

data.diagnosis=[1 if each == 'M' else 0 for each in data.diagnosis]
data.drop(['id','Unnamed: 32'],axis=1,inplace=True)
x_data=data.drop(['diagnosis'],axis=1)
y=data.diagnosis.values

# %% Normalization

x=(x_data- np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

# %% Train Test Split
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T



# %% initialize parameters

# data dimension's equal 30

def initialize_weight_bias(dimension):
    weight=np.full((dimension,1),0.01)
    bias=0.0
    return weight,bias

def sigmoid(z):
    y_head=1/(1+np.exp(-z))
    return y_head

def forward_backward_propagation(w,b,x_train,y_train):
    #forward propagation
    z=np.dot(w.T,x_train)+b
    y_head=sigmoid(z)
    loss=-y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost= (np.sum(loss))/ x_train.shape[1]
    
    #backward propagation
    derivative_weight=(np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias=np.sum(y_head-y_train)/x_train.shape[1]
    gradients={"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    
    return cost,gradients
    
    
def update(w,b,x_train,y_train,learning_rate,NumberOf_iteration):
    cost_list=[]
    cost_list2=[]
    index=[]
    
    for i in range(NumberOf_iteration):
        cost,gradients= forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        
        w=w-learning_rate * gradients["derivative_weight"]
        b=b- learning_rate* gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i : %f"%(i,cost))
     
    parameters={ "weight": w , "bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of iteration")
    plt.ylabel("cost")
    plt.show()
    
    return parameters,gradients,cost_list
#%% Prediction 
def predict(w,b,x_test):
    z=sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction= np.zeros((1,x_test.shape[1]))
    
    for i in range(z.shape[1]):
        if z[0,i] <=0.5:
            Y_prediction[0,i]=0
        else:
            Y_prediction[0,i]=1
            
    return Y_prediction            
    
    
def logistig_Regression(x_train, y_train,x_test,y_test,learning_Rate,num_iteration):
    dimension=x_train.shape[0]
    w,b=initialize_weight_bias(dimension)
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_Rate,num_iteration)
   # parameters , gradients, cost_list= update(w,b,x_train,y_train,learning_Rate,num_iteration)
    y_prediction_test=predict(parameters["weight"],parameters["bias"],x_test)
    
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
 
logistig_Regression(x_train,y_train,x_test,y_test,learning_Rate=3,num_iteration=80)
    

##%%
#
#from sklearn.linear_model import LogisticRegression
#
#Lr=LogisticRegression(max_iter=300)
#Lr.fit(x_train.T,y_train.T)
#print(Lr.score(x_test.T,y_test.T))
#
#
#


















    
    
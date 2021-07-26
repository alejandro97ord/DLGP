from dlgp import dlgp
from hypOp import hypOp

import time
from scipy.io import loadmat
import numpy as np
import random
from math import log

print("Select")
select = 1#int(input())


if select == 1:                
    data = loadmat("..\cpp\Real_Sarcos_long.mat")
    hyps = loadmat("..\cpp\hyps_Real_Sarcos_long.mat")
    
    X_train = data['X_train'][:,:].transpose()
    Y_train = data['Y_train'][:,:].transpose()
    
    X_test = data['X_test'][:,:].transpose()
    Y_test = data['Y_test'][:,:].transpose()
    
if select == 2:
    data2= loadmat(r"..\..\uci\buzz\buzz.mat")
    data2 = data2['data'][:,:]
    
    X_train = data2[:,0:77].transpose()
    Y_train = data2[:,77].transpose()

if select == 3 :
    data2= loadmat(r"..\..\uci\houseelectric\houseelectric.mat")
    data2 = data2['data'][:,:]
    
    X_train = data2[:,:].transpose()
    Y_train = data2[:,5].transpose()
    #data in line 5 is output:
    X_train = np.delete(X_train,5, axis = 0)

#determine number of ouputs
ins = X_train.shape[0]
outs = Y_train.shape[0]
if Y_train.ndim == 1:
    outs = 1
    Y_train = np.reshape(Y_train,[1, Y_train.shape[0]])
    
#initialize DLGP

 #(xDimensionality , outputs , pts , max. number of leaves , ard)
gp01 = dlgp(ins,outs,50,50000,True)  
gp01.wo = 100000
'''
ptsHyp = 800 #points used for hyperparameter optimization
hypOp( ptsHyp , 100 , X_train[:,0:ptsHyp] , Y_train[:, 0:ptsHyp]) #dataset, pts, iterations
print("Press enter to continue")
input()
'''
hyps = loadmat('/Users/alejandro/Desktop/P10/Datasets/hyps_Real_Sarcos_long.mat')
gp01.sigmaF = np.transpose(hyps['sigf'])
gp01.sigmaN = np.transpose(hyps['sign'])
gp01.lengthS = hyps['ls']


'''
#train first points
print("Initial training")
for k in range(ptsHyp):
    gp01.update(X_train[:,k], Y_train[:,k])
'''

output = np.zeros( [outs , X_test.shape[1]] , dtype = float) 
error = np.zeros( [outs , X_test.shape[1]] , dtype = float)

print("Starting test:")
a = time.time()
#for j in range(ptsHyp,X_train.shape[1]):
for j in range(X_train.shape[1]):
    if j%1000 == 0:
        print(j)
    #output[ : , j ] = gp01.predict( X_train[:,j] )
    
    gp01.update(X_train[:,j], Y_train[:,j])
a0 = gp01.alpha[:,0:5000]
    
for k in range(X_test.shape[1]):
    output[:,k] = gp01.predict(X_test[:,k])
error = (output - Y_test ) ** 2
error = np.mean( error ,1)
error = error / np.var(Y_test, axis = 1)
    
#cumulative error
#error = np.cumsum(error, axis = 1) / (np.array( range(Y_train.shape[1]) ) + 1 )

b = time.time()  

c = time.time()
print(c-a)
print(error)
# DLGP without ARD is still not supported
# total time with no aho is ca. 39 s
# agregar aho opcional , ard , mean function , composite kernel , 
#0.2900    0.6642    0.4324    0.8989    0.3151    0.2478    0.6650

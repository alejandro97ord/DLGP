from dlgp import dlgp
from hypOp import hypOp

import time
from scipy.io import loadmat
import numpy as np
import random
from math import log

print("Select")
select = 2#int(input())


if select == 1:                
    data = loadmat("..\cpp\Real_Sarcos_long.mat")
    hyps = loadmat("..\cpp\hyps_Real_Sarcos_long.mat")
    
    X_train = data['X_train'][:,:].transpose()
    Y_train = data['Y_train'][:,:].transpose()
    
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

ptsHyp = 801 #points used for hyperparameter optimization
hypOp( ptsHyp , 100 , X_train[:,0:ptsHyp] , Y_train[:, 0:ptsHyp]) #dataset, pts, iterations
print("Press enter to continue")
input()
hyps = loadmat("hyps.mat")
gp01.sigmaF = hyps['sf']
gp01.sigmaN = hyps['sn']
gp01.lengthS = hyps['L']

gp01.wo = 300

#train first points
print("Initial training")
for k in range(ptsHyp):
    gp01.update(X_train[:,k], Y_train[:,k])


output = np.zeros( [outs , X_train.shape[1]] , dtype = float) 
error = np.zeros( [outs , X_train.shape[1]] , dtype = float)

print("Starting test:")
a = time.time()
for j in range(ptsHyp,X_train.shape[1]):
    if j%1000 == 0:
        print(j)
    output[ : , j ] = gp01.predict( X_train[:,j] )
    
    gp01.update(X_train[:,j], Y_train[:,j])
    
    error[ : , j ] = ( (output[ : , j ] - Y_train[ : , j ] ) ** 2 ) / Y_train.var()
    
#cumulative error
error = np.cumsum(error, axis = 1) / (np.array( range(Y_train.shape[1]) ) + 1 )

b = time.time()  

c = time.time()
print(c-a)
print(error[:,-1])
# DLGP without ARD is still not supported
# total time with no aho is ca. 39 s
# agregar aho opcional , ard , mean function , composite kernel , 
#0.0068      0.0036      0.0029      0.0008      0.0094      0.0059      0.0018
#0.00607065, 0.00196677, 0.00175993, 0.00032937, 0.00854694, 0.00485393, 0.0012706

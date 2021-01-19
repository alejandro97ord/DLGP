from dlgp import dlgp
from hypOp import hypOp

import time
from scipy.io import loadmat
import numpy as np
import random

print("Select")
select = int(input())


if select == 1:                
    data = loadmat("..\cpp\Real_Sarcos_long.mat")
    #hyps = loadmat("..\cpp\hyps500ptsX100it.mat")
    hyps = loadmat("..\cpp\hyps_Real_Sarcos_long.mat")    
    
    outs = 7

    gp01.wo = 300
    
    X_train = np.transpose(data['X_train'][:,:] )
    Y_train = np.transpose( data['Y_train'][:,:] )
    
if select == 2:
    data2= loadmat(r"..\..\uci\buzz\buzz.mat")
    data2 = data2['data'][:,:]
    
    outs = 1
    
    X_train = data2[:,0:77].transpose()
    Y_train = data2[:,77].transpose()

if select == 3 :
    data2= loadmat(r"..\..\uci\houseelectric\houseelectric.mat")
    data2 = data2['data'][:,:]
    outs = 1
    
    X_train = data2[:,:].transpose()
    Y_train = data2[:,5].transpose()
    #data in line 5 is output:
    X_train = np.delete(X_train,5, axis = 0)
    

gp01 = dlgp(77,outs,50,50000,True)   #(xDimensionality , outputs , pts , max. number of leaves , ard)
hyps = loadmat("hyps.mat")
gp01.sigmaF = hyps['sf'][0,:]
gp01.sigmaN = hyps['sn'][0,:]
gp01.lengthS = hyps['L']
gp01.wo = 300

hypOp(select, 1000,150) #dataset, pts, iterations
    
if Y_train.ndim == 1:
    Y_train = np.reshape(Y_train,[1, Y_train.shape[0]])


output = np.zeros( [outs , X_train.shape[1]] , dtype = float) 
error = np.zeros( [outs , X_train.shape[1]] , dtype = float)

a = time.time()
for j in range(X_train.shape[1]):
    output[ : , j ] = gp01.predict( X_train[:,j] )
    
    gp01.update(X_train[:,j], Y_train[:,j])
    
    error[ : , j ] = ( (output[ : , j ] - Y_train[ : , j ] ) ** 2 ) / Y_train.var()
    

error = np.cumsum(error, axis = 1) / (np.array( range(Y_train.shape[1]) ) + 1 )

b = time.time()  

c = time.time()
print(c-a)

 # total time with no aho is ca. 39 s
 # agregar aho opcional , ard , mean function , composite kernel , 
 #0.0068      0.0036      0.0029      0.0008      0.0094      0.0059      0.0018
 #0.00607065, 0.00196677, 0.00175993, 0.00032937, 0.00854694, 0.00485393, 0.0012706

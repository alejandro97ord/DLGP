import numpy
from scipy.io import loadmat
from time import time

def tkernel(Xi, Xj, sigmaF, lengthS):
    kern = numpy.empty(Xi.shape[0] ,dtype = float)
    for i in range(Xi.shape[0]):
        kern[i] =  (sigmaF**2) * numpy.exp( -0.5* numpy.sum(((Xi[i,:]- Xj)/lengthS)**2)) 
    return kern

pts = 100;

data = loadmat("..\cpp\Real_Sarcos_long.mat")
hyps = loadmat("..\cpp\hyps_Real_Sarcos_long.mat")

X_train = data['X_train'][0:pts,:]
y = data['Y_train'][0:pts,0]

ls = hyps['ls']
sigmaF = hyps['sigf']
talp = numpy.empty([pts] , dtype = float)
for j in range(pts): # de 0 a 99
    K = []
    K = numpy.empty([j+1,j+1] , dtype = float)
    for r in range(j+1): 
        rk = tkernel( X_train[0:j+1,:] , X_train[r,:] , sigmaF[0,0] , ls[:,0] )  
        K[r,:] = numpy.transpose(rk)        
    start = time()
    for p in range(100):
        alpha = numpy.linalg.solve(K,y[0:j+1])
        del alpha
    end = time()
    talp[j] = (end - start)/100



print("DONE")
        
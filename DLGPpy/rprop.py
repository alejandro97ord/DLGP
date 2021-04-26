import numpy as np


class rpropOnline:
    def __init__(self, X, Y, its):
        #data
        self.X = X
        self.Y = Y 
        xSize = X.shape[0]
        
        #set initial delta
        self.delta = 0.1 + np.zeros([xSize+2, 1], dtype = float)
        
        #get first gradient
        #let user set first guess?
        self.x0 = np.ones([xSize+2,1], dtype = float)
        self.dlik0 = self.grad(self.x0)

        for j in range(its):
            self.PR()
            #print(self.dlik0)
        
        
        
    def PR(self):
        #rprop parameters
        dmax = 50
        dmin = 1e-6
        etap = 1.2
        etam = 0.5
        #new hyperparameters
        self.x0  = self.x0 + np.sign( self.dlik0 ) * self.delta
        #new gradient and delta
        dfx1 = self.grad( self.x0  )
        d1 = (self.dlik0 * dfx1 > 0).astype(int)* self.delta * etap + \
            (self.dlik0 * dfx1 < 0).astype(int)* self.delta * etam + \
                (self.dlik0 * dfx1 == 0).astype(int)* self.delta 
        d1 = d1 * (d1 >= dmin).astype(int) * (d1 <= dmax).astype(int)+\
            (d1 < dmin).astype(int)*dmin + (d1 > dmax).astype(int)*dmax
        #store new data
        self.dlik0 = dfx1
        self.delta = d1
        #self.x0 = x1
            
    
    
    def grad(self, x):
        #gradient
        df = x*0#np.zeros([x.shape[]] , dtype = float)
        #obtain covaraince matrix and prediction vector
        dim = self.X.shape[1]
        Ki = self.fkernel(self.X)
        Kn = Ki + np.identity(dim) * (x[1,:]**2)
        alpha = np.zeros([dim,1], dtype = float)
        alpha[:,0] = np.linalg.solve(Kn , self.Y)
        #get gradient values
        auxDer = np.dot(alpha , alpha.transpose()) - np.linalg.inv(Kn)
        df[0] = 0.5 * np.trace( np.dot(auxDer , (Ki * 2)/x[0,:] ))
        df[1] = 0.5 * np.trace( np.dot(auxDer , np.identity(dim)* 2 * x[1,:] ))
        for i in range(x.shape[0]-2):
            k1,k2 = np.meshgrid(self.X[i,:] , self.X[i,:])
            df[2+i] = 0.5 * np.trace( np.dot(auxDer , np.dot(Ki , ((k1-k2)**2 / x[2+i,:]**3 ))))
        f = -0.5 * np.dot(alpha.transpose() , self.Y)  - 0.5 * np.log(np.linalg.det(Kn)) -\
            0.5* dim * np.log(2 * np.pi)
        print(f)
        return df
    
    def fkernel(self, Xi):
        kern = np.zeros( [Xi.shape[1]  , Xi.shape[1] ] , dtype = float )
        for p in range(Xi.shape[0]):
            k1,k2 = np.meshgrid( Xi[p,:] , Xi[p,:] )
            kern = kern + (k1-k2)**2 / (self.x0[2+p,:]**2)
        return (self.x0[0,:] ** 2) * (np.e ** (-0.5 * kern))
                        
        
        
        
        
        
        
        
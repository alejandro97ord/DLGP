import numpy as np
import random


class dlgp:
    def __init__(self, xSize, outs, pts, N, ard):
        #user defined values
        self.pts = pts #data limit per local GP
        self.N = N #max. number of leaves
        self.xSize = xSize #dimensionality of X
        self.outs = outs #dimensionality of Y
        #parameters
        self.wo = 300 #factor width / overlapping
        
        #auto. hyp. optimization
        self.sigmaF = np.ones([outs, N] , dtype = float)
        self.sigmaN = np.ones([outs, N] , dtype = float)
        
        if ard:
            self.lengthS = np.ones([xSize * outs , N ] ,dtype = float)
            self.amountL = xSize
            #hyperparameter optimization variables
            self.dlik0 = 0.1 * np.ones([(xSize + 2) * outs, N], dtype = float) #local gradients
            self.delta = np.ones([(xSize + 2) * outs, N], dtype = float) #local stepsize
        else:
            self.lengthS = 1
            self.amountL = 1
        #data
        self.X = np.zeros( [xSize , pts * N] , dtype = float )
        self.Y = np.zeros( [outs , pts * N] , dtype = float )
        self.K = np.zeros( [pts*outs , pts * N] , dtype = float ) #covariance matrix
        self.alpha = np.zeros( [pts*outs , N] , dtype = float ) #prediction vectors
        #aux variables
        self.count = 0 #number of leaves - 1
        self.localCount = np.zeros( 2 * N - 1 , dtype = int ) #pts in each set
        self.medians = np.zeros( 2 * N - 1 , dtype = float ) #vector of hyperplanes
        self.overlapD = np.zeros( 2 * N - 1 , dtype = int) #overlap dimension
        self.overlapW = np.zeros( 2 * N - 1 , dtype = float) #overlap width
        self.auxUbic = np.zeros( 2 * N - 1 , dtype = int ) - 1 #map the position of data
        self.auxUbic[0] = 0
        self.children = np.zeros( [2, 2 * N - 1] , dtype = int) -1 #line 1: left child, line 2: right child

        
    def kernel(self, Xi, Xj, out): #returns a scalar or a vector
        if Xi.ndim == 1:
            kern =  (self.sigmaF[out]**2) * np.exp( -0.5* np.sum(((Xi - Xj)/self.lengthS[:,out])**2)) 
            return kern
        else:
            kern = (self.sigmaF[out]**2) * np.exp(-0.5*np.sum(((Xi.transpose() - Xj)/self.lengthS[:,out])**2, axis = 1))
            return kern
        
    def fkernel(Xi, hyp): #returns a kernel matrix of Xi and itself
    #hyp is [sigmaN, sigmaN, sigmaL_1, sigmaL_2, ... , sigmaL_xSize]
        kern = np.zeros( [ Xi.shape[1] , Xi.shape[1] ], dtype = float)
        if hyp.shape[0] > 3:
            for p in range (hyp.shape[0] - 2):
                k1, _ = np.meshgrid(Xi[p, :], Xi[p, :])
                print(p)
                kern = (kern + ( k1 - np.transpose(k1) )**2 ) / (hyp[p+1]**2)
        else:
            for p in range(Xi.shape[0]):
                k1, _ = np.meshgrid(Xi[p, :], Xi[p, :])
                print(k1)
                kern = (kern + ( k1 - np.transpose(k1) )**2 ) / (hyp[2]**2)
        kern = (hyp[0]**2) * np.e ** (-0.5 * kern)
        return kern
    
    def rprop(self, model):
        for p in range(self.outs):
            #get previous hyperparameters
            x0 = np.append( self.sigmaF[p+1 , self.auxUbic[model] ] ,   \
                           self.sigmaF[p+1 , self.auxUbic[model] ] , \
                               self.lengthS[ p * self.amountL + 1 : (p+1) * self.amountL , \
                                            self.auxUbic[model]])
            dmax = 50
            dmin = 1e-6
            etap = 1.2
            etam = 0.5
            #get new hyperparameters
            x1 = x0 + np.sign( self.dlik0[ p*(self.amountL+2) + 1 : \
                        (p+1)*(self.amountL+2) , self.auxUbic[model]  ] * \
                              self.delta[ p*(self.amountL+2)+1:\
                                    (p+1)*(self.amountL+2),self.auxUbic[model]])
            #sigmaN and sigmaF value restricions
            if abs(x1[1])  < 0.01:
                x1[1] = np.sign(x1[1])*.01
            if np.abs(x1[0] / x1[1]) > 1000:
                x1[0] = np.sign(x1[0]) * x1[1] * 1000
            #get the gradient    
            dfx1 = self.grad(x1, model, p)
            #update delta and check if it fulfills requirements
            dfx01 = self.dlik0[p*(self.amountL+2)+1 : (p+1)*(self.amountL+2),\
                self.auxUbic[model]]*dfx1 #product of gradients
            delta0 = self.delta[ p*(self.amountL+2)+1 : (p+1)*(self.amountL+2) \
                ,self.auxUbic[model]] #previous delta
            d1 = (dfx01>0)*delta0*etap + (dfx01<0)*delta0*etam + (dfx01==0)*delta0*etap
            d1 = d1*(d1>=dmin & d1<=dmax) + (d1<dmin)*dmin + (d1>dmax)*dmax
            
            #update properties
            self.dlik0[p*(self.amountL+2)+1 : (p+1)*(self.amountL+2),\
                self.auxUbic[model]] = dfx1
            self.delta[ p*(self.amountL+2)+1 : (p+1)*(self.amountL+2) \
                ,self.auxUbic[model]] = d1
            self.sigmaF[p+1, self.auxUbic[model]] = x1[0]
            self.sigmaN[p+1, self.auxUbic[model]] = x1[1]
            self.lengthS[p*self.amountL+1:(p+1)*self.amountL, self.auxUbic[model]]=\
                x1[2:]
                
    def updateParam(self, x, model):
        self.localCount[model] += 1
        pos = self.auxUbic[model]
        if self.localCount[model] == 1:
            for p in range(self.outs):
                kVal = self.kernel(x,x,p)
                self.K[ p*self.pts , pos*self.pts ] = kVal + self.sigmaN[p]**2
                self.alpha[ p * self.pts , pos ] =  self.Y[ p,  pos * self.pts ] /kVal
        else:
            auxX = self.X[ : , pos*self.pts : pos * self.pts + self.localCount[model]  ] #does not include x
            auxY = self.Y[ : , pos*self.pts : pos * self.pts + self.localCount[model]  ]
            for p in range(self.outs):
                b = self.kernel( auxX , x, p )
                b[-1] += self.sigmaN[p] ** 2
                auxOut = p * self.pts
                
                self.K[auxOut + self.localCount[model] - 1 ,
                       pos * self.pts : pos * self.pts + self.localCount[model] -1] = b[0:-1]
                self.K[ auxOut : auxOut + self.localCount[model] ,
                       pos * self.pts + self.localCount[model] -1 ] = np.transpose(b)
                
                self.alpha[ auxOut : auxOut + self.localCount[model] ,  pos ] = \
                    np.linalg.solve( self.K[ auxOut : auxOut + self.localCount[model] ,
                           pos*self.pts : pos * self.pts + self.localCount[model] ] , auxY[ p , : ] )

    def addPoint(self, x, y,model):
        if self.localCount[model] < self.pts:
            self.X[ : , self.auxUbic[model] * self.pts + self.localCount[model] ] = x 
            self.Y[ : , self.auxUbic[model] * self.pts + self.localCount[model] ] = y 
            self.updateParam(x , model)
        if self.localCount[model] == self.pts:
            self.divide(model)
    
    def divide(self, model):
        if self.auxUbic[-1] != -1:
            print( "no room for more divisions" )
            return
        #compute widths in all dimensions
        width = self.X[ : , self.auxUbic[ model ] * self.pts : self.auxUbic[ model ] *
            self.pts + self.pts ].max(axis = 1) - self.X[ : , self.auxUbic[ model ] *
                self.pts : self.auxUbic[ model ] * self.pts + self.pts ].min(axis = 1)
                                                             
        #obtain cutting dimension
        cutD = np.argmax(width)
        width = width.max()
        #compute hyperplane
        mP = (self.X[ cutD , self.auxUbic[ model ] * self.pts : self.auxUbic[ model ] * 
            self.pts + self.pts].max() + self.X[ cutD , self.auxUbic[ model ] *
                self.pts : self.auxUbic[ model ] * self.pts + self.pts ].min())/2
        
        #get overlapping region
        o = width / self.wo
        if o == 0:
            o = 0.1
        
        self.medians[model] = mP  #set model hyperplane
        self.overlapD[model] = cutD  #cut dimension
        self.overlapW[model] = o  #width of overlap
        
        xL = np.zeros( [self.xSize, self.pts] , dtype = float )
        xR = np.zeros( [self.xSize, self.pts] , dtype = float )
        yL = np.zeros( [self.outs, self.pts] , dtype = float )
        yR = np.zeros( [self.outs, self.pts] , dtype = float )
        
        lcount = 0
        rcount = 0
        
        iL = np.zeros( self.pts, dtype = int )
        iR = np.zeros( self.pts, dtype = int )
        
        for i in range(self.pts):
            xD = self.X[ cutD , self.auxUbic[model] * self.pts + i ]
            if xD < mP - 0.5 * o:
                xL[:, lcount] = self.X[:, self.auxUbic[model] * self.pts + i ]
                yL[:, lcount] = self.Y[:, self.auxUbic[model] * self.pts + i ]
                iL[lcount] = i
                lcount += 1
            elif xD >= mP - 0.5*o and xD <= mP + 0.5*o: #if in overlapping
                pL = 0.5 + (xD - mP) / o #prob. of being in left
                if pL >= random.random() and pL != 0: #left selected
                    xL[:, lcount] = self.X[:, self.auxUbic[model] * self.pts + i ]
                    yL[:, lcount] = self.Y[:, self.auxUbic[model] * self.pts + i ]
                    iL[lcount] = i
                    lcount += 1
                else:
                    xR[:, rcount] = self.X[:, self.auxUbic[model] * self.pts + i ]
                    yR[:, rcount] = self.Y[:, self.auxUbic[model] * self.pts + i ]
                    iR[rcount] = i
                    rcount += 1
            elif xD > mP + 0.5*o: #if in right
                xR[:, rcount] = self.X[:, self.auxUbic[model] * self.pts + i ]
                yR[:, rcount] = self.Y[:, self.auxUbic[model] * self.pts + i ]
                iR[rcount] = i
                rcount += 1
        self.localCount[model] = 0
        #update counter
        if self.count == 0:
            self.count+=1
        else:
            self.count+=2
        #assign children
        self.children[0,model] = self.count
        self.children[1,model] = self.count +1
        #set local count of children
        self.localCount[self.count] = lcount
        self.localCount[self.count + 1] = rcount 
        self.auxUbic[self.count] = self.auxUbic[model]
        self.auxUbic[self.count+1] = self.auxUbic.max() + 1
        #values for K permutation
        order = np.concatenate( (iL[0:lcount] , iR[0:rcount]) )
        #update parameters of child models
        for p in range(self.outs):
            newK = self.K[ p * self.pts : (p+1) * self.pts , self.auxUbic[model] * self.pts:
                self.auxUbic[model] * self.pts + self.pts]
            #permute K
            newK[ range(self.pts), : ] = newK[ order , : ]
            newK[ : , range(self.pts) ] = newK[ : , order ]
            #set child K
            self.K[p*self.pts : p * self.pts + lcount , self.auxUbic[self.count] * self.pts : 
                 self.auxUbic[self.count] * self.pts + lcount ] = newK[ 0 : lcount , 0 : lcount ]
            self.K[p*self.pts : p * self.pts + rcount  , self.auxUbic[self.count+1] * self.pts : 
                 self.auxUbic[self.count+1] * self.pts + rcount  ] = \
                    newK[lcount : self.pts,lcount : self.pts]
            #set child alpha
            self.alpha[ p * self.pts : p * self.pts + lcount , self.auxUbic[self.count]] = \
                np.linalg.solve(newK[ 0 : lcount , 0 : lcount ] , yL[p, 0:lcount].transpose())
            self.alpha[ p * self.pts : p * self.pts + rcount , self.auxUbic[self.count+1]] = \
                np.linalg.solve(newK[lcount : self.pts,lcount : self.pts], yR[p, 0:rcount].transpose())
        #parent will not have more data:
        self.auxUbic[model] = -1
        #relocate X Y to children
        self.X[: , self.auxUbic[self.count] * self.pts : 
               self.auxUbic[self.count] *self.pts + self.pts ] = xL
        self.X[: , self.auxUbic[self.count+1] * self.pts : 
               self.auxUbic[self.count+1] *self.pts + self.pts ] = xR  
        self.Y[: , self.auxUbic[self.count] * self.pts : 
               self.auxUbic[self.count] *self.pts + self.pts ] = yL
        self.Y[: , self.auxUbic[self.count+1] * self.pts : 
               self.auxUbic[self.count+1] *self.pts + self.pts ] = yR     
    
    def activation(self, x, model):
        if ( self.children[ 0 , model ] == -1 ): #return 0 if model is a leaf
            return 0
        mP = self.medians[model] #hyperplane value
        xD = x[ self.overlapD[model] ] #x value in cut dimension
        o = self.overlapW[model]    #overlap width
        if xD < mP - 0.5*o:
            return 1
        elif xD >= mP - 0.5*o  and xD <= mP + 0.5*o:
            return 0.5 + (xD - mP) / o
        else:
            return 0
        
    def update( self , x , y ):
        model = 0
        while self.children[0,model] != -1: #while model is parent
            #search for a leaf
            pL = self.activation( x , model )
            if pL >= random.random() and pL != 0:
                model = self.children[0,model] #go left
            else:
                model = self.children[1,model] #go right
        self.addPoint(x, y, model)
                
    def predict(self, x):
        out = np.zeros( self.outs , dtype = float )
        models = np.zeros( 1000 , dtype = int )
        probs = np.zeros( 1000 , dtype = float ) + 1
        
        mCount = 1
        while self.children[ 0 , models[0:mCount] ].sum() != -1 * mCount:
            for j in range( mCount ):
                if self.children[ 0 , models[j] ] != -1: #go deeper in three if node has children
                    pL = self.activation( x , models[j] )
                    if pL == 1:
                        models[j] = self.children[ 0 , models[j] ]
                    elif pL == 0:
                        models[j] = self.children[ 1 , models[j] ]
                    elif 1 > pL > 0:
                        mCount += 1
                        
                        models[ mCount - 1 ] = self.children[ 1 , models[j] ]
                        probs[ mCount - 1 ] = probs[j] * ( 1 - pL )
                        
                        models[j] = self.children[ 0 , models[j] ]
                        probs[j] = probs[j] * pL
        for p in range(self.outs):
            for i in range( mCount ):
                model = models[i]
                pred = np.dot(self.kernel( self.X[ : , self.auxUbic[model]*self.pts : \
                    self.auxUbic[model]*self.pts + self.localCount[model] ] , x , p ), 
                        self.alpha[ p*self.pts : p*self.pts + self.localCount[model] , 
                            self.auxUbic[model] ])
                out[p] += pred * probs[i]
        return out
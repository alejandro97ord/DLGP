import numpy as np
import torch
import gpytorch
from scipy.io import loadmat, savemat

def hypOp(select, amountPts, iterations):
    
    data2= loadmat(r"..\..\uci\buzz\buzz.mat")
    data2 = data2['data'][:,:]

    x = data2[0:amountPts,0:-1]
    y = data2[0:amountPts,-1]
    
    '''
    x = data2[0:100,:]
    y = data2[0:100,5]
    #y = y.reshape(-1,1)
    #data in line 5 is output:
    x = np.delete(x,5, axis = 1)
    '''
    ins = x.shape[1]
    outs = 1
    
    for p in range(outs):
    
        
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(ExactGPModel, self).__init__(x, y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ins))
        
            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(x, y, likelihood)
        
        import os
        smoke_test = ('CI' in os.environ)
        training_iter = 2 if smoke_test else iterations
        
        
        # Find optimal model hyperparameters
        model.train()
        likelihood.train()
        
        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1)
        
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(x)
            # Calc loss and backprop gradients
            loss = -mll(output, y)
            loss.backward()
            print('DoF %d Iter %d/%d - Loss: %.3f   OutputScale: %.3f   noise: %.3f' % (
                p-20,i + 1, training_iter, loss.item(),
                model.covar_module.outputscale.item()**0.5,
                model.likelihood.noise.item()**0.5
            ))
            optimizer.step()
            
        Laux =  model.covar_module.base_kernel.lengthscale[0][0].item()  
        for j in range(0,ins-1):
            Laux = np.vstack([Laux,model.covar_module.base_kernel.lengthscale[0][j].item()])
        if p == 0:
            L = Laux
            sn = model.likelihood.noise.item()**0.5
            sf = model.covar_module.outputscale.item()**0.5
        else:
            L = np.hstack([L,Laux])
            sn = np.vstack([sn,model.likelihood.noise.item()**0.5])
            sf = np.vstack([sf,model.covar_module.outputscale.item()**0.5])
        
        savemat("hyps.mat", mdict={'sf': sf, 'sn': sn, 'L': L})
        print("hyps obtained")
'''
np.savetxt('sf.csv',sf,delimiter = ",")
np.savetxt('sn.csv',sn,delimiter = ",")
np.savetxt('L.csv',L,delimiter = ",")
'''

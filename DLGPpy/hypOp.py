#based on https://docs.gpytorch.ai/en/v1.1.1/examples/01_Exact_GPs/

import numpy as np
import torch
import gpytorch
from scipy.io import loadmat, savemat

def hypOp(amountPts, iterations, x, y):
    x = x.transpose()
    y0 = y.transpose()
    ins = x.shape[1]
    outs = y.shape[0]
    x = torch.from_numpy(x)
    for p in range(outs):
        y = torch.from_numpy(y0[:,p])
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

            optimizer.step()
        print('DoF %d Iter %d/%d - Loss: %.3f   OutputScale: %.3f   noise: %.3f' % (
                p,i + 1, training_iter, loss.item(),
                model.covar_module.outputscale.item()**0.5,
                model.likelihood.noise.item()**0.5
            ))            
        Laux =  model.covar_module.base_kernel.lengthscale[0][0].item()  
        for j in range(ins-1):
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

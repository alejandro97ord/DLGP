from rprop import rpropOnline
from scipy.io import loadmat

data = loadmat("..\cpp\Real_Sarcos_long.mat")

its = 100
dataPts = 10    
X_train = data['X_train'][0:dataPts,:].transpose()
Y_train = data['Y_train'][0:dataPts,:].transpose()


hyps  = rpropOnline(X_train , Y_train[0,:] , its)
# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
    
#minimize da errores!!!

def main(X,y,Lambda = 0,comprob = False):
    (m,nFeat) = np.shape(X)
    (my,nOut) = np.shape(y)
    if comprob:
        if m != my: raise TypeError('El numero de filas de X no concuerda con el de y')
    X = np.concatenate((np.ones((m,1)),X),axis=1)
    #Definir Theta
    Theta = np.zeros((nFeat + 1,nOut)).reshape(-1)
    def coste(Theta):
        Theta = np.matrix(Theta.reshape((nFeat + 1,nOut)))
        return 1./m*(-y.transpose()*np.log(sigmoid(X*Theta))-(1.-y.transpose())*np.log(1-sigmoid(X*Theta))) + Lambda/(2.*m)*np.sum(np.multiply(Theta[:,1:],Theta[:,1:]))
    def costGrad(Theta):
        Theta = np.matrix(Theta.reshape((nFeat + 1,nOut)))
        grad = 1./m*X.transpose()*(sigmoid(X*Theta)-y)
        grad[1:,:] += Lambda/float(m)*Theta[1:,:]
        return np.asarray(grad).reshape(-1)
    res = minimize(coste, Theta, method = 'BFGS', jac=costGrad, options={'maxiter': 400,'disp':True})
    print '\n1\n'
    print res.message
    print '\n2\n'
    print res.x
    print '\n3\n'
    print res.success


def sigmoid(z):
    return 1./(1+np.exp(-z))

##X=np.matrix('1 2 3;4 5 6;7 8 9;2 4 6;1 3 5;9 8 7')
##y = np.matrix('1;0;0;1;0;1')
##main(X,y,comprob=True,Lambda = 0)
##main(X,y,comprob=True,Lambda = 1)
##main(X,y,comprob=True,Lambda = 10)

##X = np.matrix(np.genfromtxt('Xlog.txt',delimiter = '|'))
##X = X[:,1:]
##y = np.matrix(np.genfromtxt('ylog.txt',delimiter = '|')).transpose()#Datos de coursera
##main(X,y,0.,True)
##main(X,y,1.,True)
##main(X,y,10.,True)

X = np.matrix(np.genfromtxt('Xlog2.txt',delimiter = '|'))
X = X[:,1:]
y = np.matrix(np.genfromtxt('ylog2.txt',delimiter = '|')).transpose()#Datos de coursera
main(X,y,0.,True)
main(X,y,1.,True)
main(X,y,10.,True)

# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
from leer import *
#minimize da errores!!!

##def main(X,y,Lambda = 0,comprob = False):
##    (m,nFeat) = np.shape(X)
##    (my,nOut) = np.shape(y)
##    if comprob:
##        if m != my: raise TypeError('El numero de filas de X no concuerda con el de y')
##    X = np.concatenate((np.ones((m,1)),X),axis=1)
##    #Definir Theta
##    Theta = np.zeros((nFeat + 1,nOut)).reshape(-1)
##    def coste(Theta):
##        Theta = np.matrix(Theta.reshape((nFeat + 1,nOut)))
##        return 1./m*(-y.transpose()*np.log(sigmoid(X*Theta))-(1.-y.transpose())*np.log(1-sigmoid(X*Theta))) + Lambda/(2.*m)*np.sum(np.multiply(Theta[:,1:],Theta[:,1:]))
##    def costGrad(Theta):
##        Theta = np.matrix(Theta.reshape((nFeat + 1,nOut)))
##        grad = 1./m*X.transpose()*(sigmoid(X*Theta)-y)
##        grad[1:,:] += Lambda/float(m)*Theta[1:,:]
##        return np.asarray(grad).reshape(-1)
##    res = minimize(coste, Theta, method = 'BFGS', jac=costGrad, options={'maxiter': 400,'disp':True})
##    print '\n1\n'
##    print res.message
##    print '\n2\n'
##    print res.x
##    print '\n3\n'
##    print res.success


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

##X = np.matrix(np.genfromtxt('Xlog2.txt',delimiter = '|'))
##X = X[:,1:]
##y = np.matrix(np.genfromtxt('ylog2.txt',delimiter = '|')).transpose()#Datos de coursera
##main(X,y,0.,True)
##main(X,y,1.,True)
##main(X,y,10.,True)





##Regresion lineal con clases y diccionarios:
class LogReg:
    def __init__(self,X,y,Lambda = 0, umbral = 0.5):
        #Añadir Xval, Xtest, yval, ytest?
        self.X = X
        self.y = y
        self.Lambda = Lambda
        self.umbral = umbral
        self.m = len(X.keys()) #Numero de transacciones
        self.nFeat = len(X.values()[0]) #Numero de variables de entrada
        for k in self.X.keys():
            self.X[k] = [1]+self.X[k] #Añadimos unos al principio de cada transaccion
        self.my = len(y.keys()) #Numero de transacciones en la y
        if self.m != self.my:
            raise TypeError('El numero de transacciones de X no concuerda con el de y')
##        self.nOutput = len(y.values()[0]) #Numero de clases de salida. En el problema binario sera 1
##        if self.nOutput != 1:
##            raise TypeError('De momento no estoy implementado para resolver clasificaciones multiclase')
        self.init_theta = [0]*(self.nFeat+1) #theta inicial, todo ceros
        self.comparar(self.y,self.predict(self.X))
    def h(self,theta,x): #Hipotesis para valores dados de theta y x
        suma = 0
        for i in range(len(theta)):
            suma += theta[i]*x[i] 
        return sigmoid(suma)         
    def coste(self, theta, X, y, Lambda):
        suma = 0.
        keys = X.keys()
        m = len(keys)
        for k in xrange(m):
            suma += y[keys[k]]*np.log(self.h(theta,X[keys[k]]))
            suma += (1-y[keys[k]])*np.log(1-self.h(theta,X[keys[k]]))
        suma = suma/(float(m))
        suma2 = 0.
        for j in range(1,len(theta)):
            suma2 += theta[j]**2
        suma2 = suma2*Lambda/(2.*m)
        return -suma + suma2
    def grad(self, theta, X, y, Lambda):
        keys = X.keys()
        m = len(keys)
        grad = [0]*len(theta)
        for j in range(len(grad)):
            for k in xrange(m):
                grad[j] += (self.h(theta,X[keys[k]])-y[keys[k]])*X[keys[k]][j]
            if j != 0:
                grad[j] += Lambda*theta[j]
        return [x/float(m) for x in grad]
    def optim(self):
        res = minimize(self.coste, self.init_theta,
                       args = (self.X, self.y, self.Lambda),
                       method = 'BFGS', jac=self.grad,
                       options={'maxiter': 400,'disp':True})
        self.theta = res.x
    def predict(self,X):
        self.optim()
        keys = X.keys()
        m = len(keys)
        ypred = {}
        for i in xrange(m):
            if self.h(self.theta,X[keys[i]]) >= self.umbral: ypred[keys[i]] = 1
            else: ypred[keys[i]] = 0
        return ypred
    def comparar(self,y,ypred):
        keys = y.keys()
        m = len(keys)
        mpred = len(ypred.keys())
        if m != mpred:
            raise TypeError('El numero de casos de y no concuerda con el de ypred')
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for k in keys:
            if y[k] == 1:
                if ypred[k] == 1:
                    TP += 1
                else:
                    FN += 1
            else:
                if ypred[k] == 1:
                    FP += 1
                else:
                    TN += 1
        acc = float(TP + TN)/float(m)*100
        rec = float(TP)/float(TP+FN)*100
        spec = float(TN)/float(FP+TN)*100
        prec = float(TP)/float(TP+FP)*100
        f = 2*prec*rec/(prec+rec)
        g = np.sqrt(rec*spec)
        print 'Accuracy:',acc,'%\n'
        print 'Sensitivity (Recall):',rec,'%\n'
        print 'Specificity:',spec,'%\n'
        print 'Precission:',prec,'%\n'
        print 'F score:',f,'%\n'
        print 'G mean:',g,'%\n'
        
X = abrir('X_1.txt')
y = abrir('y_1.txt')
prob = LogReg(X,y,Lambda=0,umbral=0.5)
            
        

























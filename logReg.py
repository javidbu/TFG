# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
from leer import *
import time
#minimize da errores!!!

#Ver si deja de dar memory errors el codigo de numpy usando numexpr
#import numexpr as ne
#ne.evaluate('SENTENCIA A EVALUAR')

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
    '''Implementa una regresión logistica. Para llamarla, ejecu-
       tar LogReg(X,y,Lambda,umbral), donde X es un diccionario
       con listas de los valores de las variables (features), y
       es un diccionario con valores 0 o 1 y las mismas claves
       que X, Lambda es el parametro de regularizacion (para evi-
       tar el overfitting) y umbral es el umbral a partir del
       cual vamos a predecir y = 1.'''
    def __init__(self,X,y,Lambda = 0, umbral = 0.5):
        '''Inicia la regresion logistica, calcula el valor optimo
           de theta e imprime en pantalla algunos datos sobre la
           bondad de la prediccion.'''
        #Añadir Xval, Xtest, yval, ytest?
        self.X = X
        self.y = y
        self.Lambda = Lambda
        self.umbral = umbral
        self.m = len(X.keys()) #Numero de transacciones
        self.nFeat = len(X.values()[0]) #Numero de variables de entrada
        self.X = {k: [1] + self.X[k] for k in self.X.keys()} #Mas rapido...
##        for k in self.X.keys():
##            self.X[k] = [1]+self.X[k] #Añadimos unos al principio de cada transaccion
        self.my = len(y.keys()) #Numero de transacciones en la y
        if self.m != self.my:
            raise TypeError('El numero de transacciones de X no concuerda con el de y')
##        self.nOutput = len(y.values()[0]) #Numero de clases de salida. En el problema binario sera 1
##        if self.nOutput != 1:
##            raise TypeError('De momento no estoy implementado para resolver clasificaciones multiclase')
        print 'Variables de la clase creadas'
        print 'Hora actual: ' + time.strftime("%H:%M:%S")
        self.init_theta = [0]*(self.nFeat+1) #theta inicial, todo ceros
        print 'Theta inicial a cero'
        print 'Hora actual: ' + time.strftime("%H:%M:%S")
        self.comparar(self.y,self.predict(self.X))
    def h(self,theta,x): #Hipotesis para valores dados de theta y x
        '''Calcula la h_theta(x) de la regresion logistica.'''
        suma = sum(theta[i]*x[i] for i in xrange(len(theta))) #Mas rapido
##        suma = 0
##        for i in range(len(theta)): #Esto estaría mejor con vectores...
##            suma += theta[i]*x[i] 
        return sigmoid(suma)         
    def coste(self, theta, X, y, Lambda): #Tarda 2 minutos... El problema serio esta en el gradiente, pero esto tampoco es admisible...
        '''Calcula el valor de la funcion de coste para valores dados
           de theta, X, y y Lambda'''
        print 'Llamada a la funcion de coste'
        print 'Hora actual: ' + time.strftime("%H:%M:%S")
        m = len(X.keys())
        suma = sum(y[k]*np.log(self.h(theta,X[k])) if y[k] == 1
                    else (1-y[k])*np.log(1-self.h(theta,X[k])) for k in X.keys())
##        suma = 0.
##        keys = X.keys()
##        m = len(keys)
##        for k in xrange(m): #Esto iria mejor con vectores
##            suma += y[keys[k]]*np.log(self.h(theta,X[keys[k]]))#Aqui da errores, igual seria mejor que calculase las cosas con un if (ademas supongo que sera mas rapido)
##            suma += (1-y[keys[k]])*np.log(1-self.h(theta,X[keys[k]]))
        suma = suma/(float(m))
        suma2 = sum(j**2 for j in theta[1:])
##        suma2 = 0.
##        for j in range(1,len(theta)): #Regularizacion, ignoramos theta_0
##            suma2 += theta[j]**2
        suma2 = suma2*Lambda/(2.*m)
        print 'Coste calculado'
        print 'Hora actual: ' + time.strftime("%H:%M:%S")
        return -suma + suma2
    def grad(self, theta, X, y, Lambda):#Esto parece ser lo que mas tarda... 33 minutos!
        '''Calcula el gradiente de la funcion de coste para valores dados
           de theta, X, y y Lambda'''
        print 'Llamada al gradiente'
        print 'Hora actual: ' + time.strftime("%H:%M:%S")
        keys = X.keys()
        m = len(keys)
        grad = [sum((self.h(theta,X[k])-y[k])*float(X[k][j]) for k in X.keys()) + Lambda*theta[j] if j > 0 else
                sum((self.h(theta,X[k])-y[k])*float(X[k][j]) for k in X.keys()) for j in range(len(theta))]
##        grad = [0]*len(theta)
##        for j in range(len(grad)):
##            for k in xrange(m): #Hay que usar vectores y matrices!!!
##                grad[j] += (self.h(theta,X[keys[k]])-y[keys[k]])*X[keys[k]][j]
##            if j != 0:
##                grad[j] += Lambda*theta[j]
        print 'Gradiente calculado'
        print 'Hora actual: ' + time.strftime("%H:%M:%S")
        return [x/float(m) for x in grad]
    def optim(self):
        '''Llama a la funcion minimize para hallar el valor optimo de theta'''
        print 'Optimizando theta'
        print 'Hora actual: ' + time.strftime("%H:%M:%S")
        res = minimize(self.coste, self.init_theta,
                       args = (self.X, self.y, self.Lambda),
                       method = 'BFGS', jac=self.grad,
                       options={'maxiter': 400,'disp':True})
        #BFGS no funciona bien (precission loss... 0 iteraciones...)... probando con Newton-CG
        self.theta = res.x
        print 'Theta optimizada'
        print 'Hora actual: ' + time.strftime("%H:%M:%S")
    def predict(self,X):
        '''Predice el valor de y dado X'''
        self.optim()
        keys = X.keys()
        m = len(keys)
        print 'Empezando la prediccion'
        print 'Hora actual: ' + time.strftime("%H:%M:%S")
        ypred = {k: 1 if self.h(self.theta,X[k]) >= self.umbral else 0 for k in X.keys()}
##        ypred = {}
##        for i in xrange(m):
##            if self.h(self.theta,X[keys[i]]) >= self.umbral: ypred[keys[i]] = 1
##            else: ypred[keys[i]] = 0
        print 'Prediccion realizada'
        print 'Hora actual: ' + time.strftime("%H:%M:%S")
        return ypred
    def comparar(self,y,ypred):
        '''Compara los valores de y e ypred, calculando algunos factores
           que nos dicen si la prediccion es buena'''
        print 'Empezando la comparacion'
        print 'Hora actual: ' + time.strftime("%H:%M:%S")
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
        print 'Hora actual: ' + time.strftime("%H:%M:%S")

print 'Leyendo los datos...'
print 'Hora actual: ' + time.strftime("%H:%M:%S")
X = abrir('X_1.txt') #Tarda 7 minutos y medio... ¿No se puede mejorar de ninguna forma?
print 'Leido archivo X'
print 'Hora actual: ' + time.strftime("%H:%M:%S")
y = abrir('y_1.txt') #Tarda 18 segundos, por fin algo normal!
print 'Leido archivo y'
print 'Instanciando la clase'
print 'Hora actual: ' + time.strftime("%H:%M:%S")
prob = LogReg(X,y,Lambda=10,umbral=0.5)
            
        

























# -*- coding: utf-8 -*-
import numpy as np
from random import *
from scipy.optimize import minimize

def __main__(X, y, nLayers = 2, nInput = None, s = None, comprob = False, Lambda = 0):
    '''Hace una red neuronal de NLAYERS capas, con los datos de training (X,Y),
    con NINPUT variables en los datos de entrada. S es una lista del numero de unidades
    en cada layer.
    X tiene que tener cada training example en una fila.
    Y tiene que ser una matriz con cada training example en una fila de tipo (0,...,1,...,0)
    donde el 1 esta en la posicion de la clase a la que corresponde. Para clasificacion binaria
    basta que sea una matriz de m columnas de ceros y unos.
    Si NLAYERS, NINPUT y S no se indican, se suponen dos capas en la red.
    Si COMPROB es True se comprueba que las dimensiones de los argumentos sean las correctas.'''
    ## Variables utiles
    (m,nInputX) = np.shape(X)
    (my,nOutput) = np.shape(y)
    if nInput is None:
        nInput = nInputX
    if s is None:
        s = [nInput,nOutput]
    ## Comprobaciones (si procede)
    if comprob:
        if nInput != nInputX:
            raise TypeError('La matriz X no tiene el numero de columnas adecuado!')
        if my != m:
            raise TypeError('El numero de filas de X e y tiene que ser el mismo!')
        if len(s) != nLayers:
            raise TypeError('la longitud de nLayers no coincide con la de s!')
        if s[0] != nInput:
            raise TypeError('s[0] no coincide con nInput!')
        if s[-1] != nOutput:
            raise TypeError('s[-1] no es igual al numero de columnas de y!')
    ## Inicializacion de las Thetas
    Theta = {} #Diccionario donde se van guardando las thetas
    for l in range(1,nLayers): #Va desde 1 hasta nLayers-1
        epsilon = np.sqrt(6)/np.sqrt(s[l-1]+s[l])
        Theta[l] = np.zeros((s[l],s[l-1]+1)) #matrices de ceros de s[l] filas y s[l-1]+1 columnas
        for x in range(np.shape(Theta[l])[0]):
            for Y in range(np.shape(Theta[l])[1]):
                Theta[l][x,Y] = random()*2*epsilon-epsilon #inicializa aleatoriamente las matrices theta
    param = np.array(0).reshape(-1)[0:0]
    for l in range(1,nLayers):
        param = np.concatenate((param,np.array(Theta[l]).reshape(-1,)))
    def coste(param): #Hay que hacer que le llegue una lista de valores de theta y que luego los monte en matrices
        '''Devuelve el valor de la función de coste para valores dados de PARAM.
        PARAM debe ser un array de parametros, no un diccionario de matrices'''
        Theta = {}
        for l in range(1,nLayers):
            Theta[l] = np.matrix(param[:s[l]*(s[l-1]+1)]).reshape((s[l],s[l-1]+1))
            param = param[s[l]*(s[l-1]+1):]
        a = {1:X}
        z = {}
        for l in range(2,nLayers+1):
            a[l-1] = np.c_[np.ones(np.shape(a[l-1])[0]),a[l-1]]
            a[l]=sigmoid(a[l-1]*Theta[l-1].transpose())
        J = -1./float(m)*np.sum(np.multiply(y,np.log(a[nLayers]))+np.multiply(1-y,np.log(1-a[nLayers])))
        suma = 0.
        for l in range(1,nLayers):
            suma += np.sum(np.multiply(Theta[l][:,1:],Theta[l][:,1:]))
        J += Lambda/(2.*m)*suma
        return J
    def costeGrad(param):
        '''Devuelve el gradiente de la funcion de coste usando el metodo back propagation.
        PARAM es un array de parametros, no un diccionario de matrices.
        Devuelve un array con los valores del gradiente.'''
        Theta = {}
        for l in range(1,nLayers):
            Theta[l] = np.matrix(param[:s[l]*(s[l-1]+1)]).reshape((s[l],s[l-1]+1))
            param = param[s[l]*(s[l-1]+1):]
        ## Back Propagation
        delta = {}
        Delta = {}
        for l in range(1,nLayers): #Va desde 1 hasta nLayers-1
            Delta[l] = np.zeros((s[l],s[l-1]+1))
        for t in range(m):
            at = {1:X[t,:]}
            for l in range(2,nLayers+1):
                at[l-1] = np.c_[np.ones(np.shape(at[l-1])[0]),at[l-1]]
                at[l]=sigmoid(at[l-1]*Theta[l-1].transpose())
            delta[nLayers] = at[nLayers]-y[t,:]
            for l in range(nLayers-1,0,-1):
                if l == nLayers-1:
                    delta[l] = np.multiply(np.multiply(delta[l+1]*Theta[l],at[l]),(1-at[l]))
                else:
                    delta[l] = np.multiply(np.multiply(delta[l+1][:,1:]*Theta[l],at[l]),(1-at[l]))
            for l in range(1,nLayers):
                if l == nLayers-1:
                    Delta[l] = Delta[l] + delta[l+1].transpose()*at[l]
                else:
                    Delta[l] = Delta[l] + (delta[l+1][:,1:]).transpose()*at[l]
        D={}
        for l in range(1,nLayers):
            D[l] = Delta[l]/float(m)
            D[l][:,1:] = D[l][:,1:] + float(Lambda)/float(m)*Theta[l][:,1:] #Revisar si aquí se divide también entre m
        grad = np.array(0).reshape(-1)[0:0]
        for l in range(1,nLayers):
            grad = np.concatenate((grad,np.array(D[l]).reshape(-1,)))
        return grad
    ## Minimizacion
    res = minimize(coste, param, method = 'BFGS', jac = costeGrad, options = {'disp':True}) #'maxiter':1 tambien da Memory Error...
    #Memory Error para BFGS, CG y Newton-CG tardan demasiado...
    print res.x
    print '\nCoste:\n',coste(res.x),'\n'
    

def sigmoid(z):
    return 1./(1+np.exp(-z))

##X=np.matrix('1 2 3;4 5 6;7 8 9;2 4 6;1 3 5;9 8 7')
##y = np.matrix('1;0;0;1;0;1')
##__main__(X,y,comprob=True,Lambda = 0)
##__main__(X,y,comprob=True,Lambda = 1)
##__main__(X,y,comprob=True,Lambda = 10)

X = np.matrix(np.genfromtxt('X.txt',delimiter = '|'))
y = np.matrix(np.genfromtxt('y.txt',delimiter = '|'))#Datos de reconocimiento de digitos
__main__(X,y,3,400,[400,25,10],True,10)

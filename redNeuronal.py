# -*- coding: utf-8 -*-
import numpy as np
from random import *

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
        Theta[l] = np.zeros((s[l],s[l-1]+1)) #np.matrix((('0 '*(s[l-1]+1)+';')*s[l])[:-2],dtype=float) #matrices de ceros de s[l+1] filas y s[l]+1 columnas
        for x in range(np.shape(Theta[l])[0]):
            for y in range(np.shape(Theta[l])[1]):
                Theta[l][x,y] = random()*2*epsilon-epsilon #inicializa aleatoriamente las matrices theta
    def coste():
        ## Calculo de la primera ronda
        a = {1:X}
        z = {}
        for l in range(2,nLayers+1):
            a[l-1] = np.c_[np.ones(np.shape(a[l-1])[0]),a[l-1]]
            a[l]=sigmoid(a[l-1]*Theta[l-1].transpose())
        ## Back Propagation
        ## Esto tarda un poco...
        delta = {}
        Delta = {}
        for l in range(1,nLayers): #Va desde 1 hasta nLayers-1
            Delta[l] = np.matrix((('0 '*(s[l-1]+1)+';')*s[l])[:-2],dtype=float)
        for t in range(m):
            at = {1:X[t,:]}
            for l in range(2,nLayers+1):
                at[l-1] = np.c_[np.ones(np.shape(at[l-1])[0]),at[l-1]]
                at[l]=sigmoid(at[l-1]*Theta[l-1].transpose())
            delta[nLayers] = at[nLayers]-y
            for l in range(nLayers-1,0,-1):
                delta[l] = np.multiply(np.multiply(Theta[l].transpose()*delta[l+1],at[l]),(1-at[l]))
            for l in range(1,nLayers):
                Delta[l] = Delta[l] + delta[l+1]*at[l]
        D={}
        for l in range(1,nLayers):
            D[l] = Delta[l]/m
            D[l][:,1:] += Lambda*Theta[l][:,1:]
        print D
    coste()
    print 'Done!'

def sigmoid(z):
    return 1./(1+np.exp(-z))

X=np.matrix('1 2 3;4 5 6;7 8 9;2 4 6;1 3 5;9 8 7')
y = np.matrix('1;0;0;1;0;1')
__main__(X,y)

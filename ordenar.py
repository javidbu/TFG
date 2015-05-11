import numpy as np
#from sklearn.datasets import load_digits

def orden(X, col):
    '''Ordena la matriz X colocando los valores de la columna col con mayor
       cociente entre la proporcion de unos y la de ceros de la ultima co-
       lumna de X. Devuelve la matriz X ordenada.'''
    frecPos = {}
    frecNeg = {}
    P = 0
    N = 0
    y = np.array(X[:,-1]).reshape((-1,)).tolist()
    for i in xrange(len(y)):
        frecPos[X[i,col]] = frecPos.get(X[i,col],0)
        frecNeg[X[i,col]] = frecNeg.get(X[i,col],0)
        if y[i] == 0:
            N += 1
            frecNeg[X[i,col]] += 1
        else:
            P += 1
            frecPos[X[i,col]] += 1
    fracc = {}
    for k in frecPos.keys():
        frecPos[k] /= float(P)
        frecNeg[k] /= float(N)
        fracc[k] = frecPos[k]/float(frecNeg[k]) if float(frecNeg[k]) != 0 else 1000
    ind = sorted(fracc.items(),key = lambda a: a[1], reverse = True)
    Xnew = X[:0,:]
    for i in ind:
        a = np.array(X[:,col]==i[0]).reshape(-1)
        Xnew = np.concatenate((Xnew,X[a]))
    return Xnew

def reducir(X,per):
    '''Devuelve las primeras filas de X correspondientes al porcentaje
       per del total de filas'''
    (m,n) = X.shape
    i = int(per*m/100.)
    return X[:i,:]

#X = load_digits(2)
#X = np.concatenate((X.data,X.target.reshape((-1,1))),axis = 1)
#Xord = orden(X,2)
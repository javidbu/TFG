from numpy import zeros, matrix, size

def binarize(X,cols,values):
    '''Recibe una matriz X, una lista de indices de columnas cols y una lista de
       tuplas de valores values, y devuelve una matriz en la que cada columna de
       X se transforma, o bien en una columna de Xnew si el indice de la columna
       no aparece en cols, o bien una serie de columnas con unos si cada valor 
       de values para esa columna esta en la columna de X, o ceros si no. Se re-
       comienda usarlo como binarize(X,cols,categorize(X,cols)).'''
    if len(cols) != len(values):
        raise TypeError('cols (list of integers) and values (list of tuples) must have the same length')
    (m,n) = X.shape
    new = sum(len(val) for val in values)
    old = n - len(cols)
    Xnew = matrix(zeros((m,new+old)))
    j = 0
    for i in range(n):
        if i not in cols:
            Xnew[:,j] = X[:,i]
            j += 1
        else:
            c = [C for (C,x) in enumerate(cols) if x == i][0]
            for k in values[c]:
                Xnew[:,j] = (X[:,i] == k).astype(int)
                j += 1
    return Xnew


def categorize(X,cols):
    '''Devuelve una lista de tuplas, cada tupla contiene los valores que toman
       las categorias en las columnas cols de X'''
    l = []
    for i in range(len(cols)):
        l.append(())
        colX = X[:,cols[i]]
        while True:
            a = colX[0,0]
            l[i] += (a,)
            colX = colX[colX[:,:] != a]
            if size(colX) == 0: break
    return l











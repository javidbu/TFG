# -*- coding: cp1252 -*-


def gradDesc(w0, f, df, Lambda = 0,
             alfa = 0.1, tol = 0.0001,
             args = {}, It = 100,
             Disp = False, DispRes = True):
    '''Lleva a cabo el algoritmo del gradient descent para una funcion dada.'''
    i = 0
    flag = None
    costAnt = f(w0) #No se como llamar a los argumentos extra que necesitan las funciones...
    while i < It:
        cost = f(w0)
        if costAnt <= cost and i != 0:
            flag = 4
            break
        if abs(cost-costAnt) < tol and i != 0:
            flag = 0
            break
        costAnt = cost
        grad = df(w0)
        if sum(g**2 for g in grad) < tol:
            flag = 1
            break
        w0 = [w0[j] - alfa*grad[j] for j in range(len(w0))]
        if Disp: print 'Iteración: %4i   Coste: %4e' % (i,cost)
        i += 1
    if flag == None:
        flag = 2
    cost = f(w0)
    if Disp: print 'Iteración: %4i   Coste: %4e' % (i,cost)
    flags = {0: 'La función de coste ha alcanzado la tolerancia.',
             1: 'El gradiente ha alcanzado la tolerancia.',
             2: 'Se ha alcanzado el número máximo de iteraciones.',
             4: 'Error! El coste está ascendiendo o se mantiene, considere introducir un valor de alfa menor.'}
    if DispRes: print flags[flag]
    if DispRes: print 'Se han realizado %i iteraciones, el coste es de %4e' % (i,cost)
    return w0
    

##gradDesc([5],lambda x:x[0]**2,lambda x:[2*x[0]],Disp=True)

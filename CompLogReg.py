from sklearn import linear_model
import numpy as np
from sklearn.datasets import load_digits
from scipy.optimize import minimize
from matplotlib.pyplot import plot, show, legend, figure

#A pesar de que minimize ha dado errores, predice igual de bien que scikit
#Resultados:
###########################################
#Parametros de mi programa
###########################################
#TP: 182
#FP: 0
#TN: 178
#FN: 0
#Accuracy: 100.0 %
#
#Sensitivity (Recall): 100.0 %
#
#Specificity: 100.0 %
#
#Precission: 100.0 %
#
#F score: 100.0 %
#
#G mean: 100.0 %
#
###########################################
#Parametros de scikit
###########################################
#TP: 182
#FP: 0
#TN: 178
#FN: 0
#Accuracy: 100.0 %
#
#Sensitivity (Recall): 100.0 %
#
#Specificity: 100.0 %
#
#Precission: 100.0 %
#
#F score: 100.0 %
#
#G mean: 100.0 %


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
    w = np.matrix(res.x).transpose()
    return np.rint(sigmoid(X*w)),sigmoid(X*w)


def sigmoid(z):
    return 1./(1+np.exp(-z))

dig = load_digits(2)
X = np.matrix(dig.data)
y = np.matrix(dig.target).transpose()
ypred,yprob1 = main(X,y,1)

TP,FP,TN,FN = 0,0,0,0
for i in range(len(y)):
    if y[i] == 1:
        if ypred[i] == 1:
            TP += 1
        else:
            FN += 1
    else:
        if ypred[i] == 1:
            FP += 1
        else:
            TN += 1
acc = float(TP + TN)/float(TP + FP + TN + FN)*100
rec = float(TP)/float(TP+FN)*100
spec = float(TN)/float(FP+TN)*100
prec = float(TP)/float(TP+FP)*100
f = 2*prec*rec/(prec+rec) if prec != 0 else 0.
g = np.sqrt(rec*spec)
print '##########################################'
print 'Parametros de mi programa'
print '##########################################'
print 'TP:',TP
print 'FP:',FP
print 'TN:',TN
print 'FN:',FN
print 'Accuracy:',acc,'%\n'
print 'Sensitivity (Recall):',rec,'%\n'
print 'Specificity:',spec,'%\n'
print 'Precission:',prec,'%\n'
print 'F score:',f,'%\n'
print 'G mean:',g,'%\n'

clf = linear_model.LogisticRegression(C=1)
clf.fit(X,y)

ypred_train = clf.predict(X)
yprob2 = clf.predict_proba(X)

TP = 0
FP = 0
TN = 0
FN = 0
for i in range(len(y)):
    if y[i] == 1:
        if ypred_train[i] == 1:
            TP += 1
        else:
            FN += 1
    else:
        if ypred_train[i] == 1:
            FP += 1
        else:
            TN += 1
acc = float(TP + TN)/float(TP + FP + TN + FN)*100
rec = float(TP)/float(TP+FN)*100
spec = float(TN)/float(FP+TN)*100
prec = float(TP)/float(TP+FP)*100
f = 2*prec*rec/(prec+rec) if prec != 0 else 0.
g = np.sqrt(rec*spec)
print '##########################################'
print 'Parametros de scikit'
print '##########################################'
print 'TP:',TP
print 'FP:',FP
print 'TN:',TN
print 'FN:',FN
print 'Accuracy:',acc,'%\n'
print 'Sensitivity (Recall):',rec,'%\n'
print 'Specificity:',spec,'%\n'
print 'Precission:',prec,'%\n'
print 'F score:',f,'%\n'
print 'G mean:',g,'%\n'


yprob2 = np.matrix(yprob2[:,1]).transpose()
print '#'*50
print 'Plotting'

err = np.multiply((yprob1-yprob2),(yprob1-yprob2)).tolist()
errAc = [sum(er[0] for er in err[:i+1]) for i in range(len(err))]


figure()
plot(yprob1,'ro',label = 'Mine')
plot(yprob2,'b',label = 'Scikit')
legend()
figure()
plot(errAc,label = 'Error acumulado')
legend(loc=2)

show()
from sklearn import linear_model
import numpy as np
from leer import guardar, abrir

print 'Cargando datos training'
X = np.matrix(np.genfromtxt('training/Split_1.txt',delimiter = '|', usecols = (2, 9, 18, 23, 27, 28,-2)))
X = np.concatenate((X,np.matrix(np.genfromtxt('training/Split_2.txt',delimiter = '|', usecols = (2, 9, 18, 23, 27, 28,-2)))))
X = np.concatenate((X,np.matrix(np.genfromtxt('training/Split_3.txt',delimiter = '|', usecols = (2, 9, 18, 23, 27, 28,-2)))))
X = np.concatenate((X,np.matrix(np.genfromtxt('training/Split_4.txt',delimiter = '|', usecols = (2, 9, 18, 23, 27, 28,-2)))))
#X = X[:10000,:]
y = np.array(X[:,-1]).reshape((-1,))
X = X[:,:-1]
print 'Datos cargados training'

print 'Cargando datos test'
X_test = np.matrix(np.genfromtxt('test/aleatorio.txt',delimiter = '|', usecols = (2, 9, 18, 23, 27, 28,-2)))
#X_test = X_test[:10000,:]
y_test = np.array(X_test[:,-1]).reshape((-1,))
X_test = X_test[:,:-1]
print 'Datos cargados test'

print 'Guardando datos'
guardar(X,'X_train.txt')
guardar(y,'y_train.txt')
guardar(X_test,'X_test.txt')
guardar(y_test,'y_test.txt')
print 'Datos guardados'

#print 'Cargando datos'
#X = abrir('X_train.txt')
#y = abrir('y_train.txt')
#X_test = abrir('X_test.txt')
#y_test = abrir('y_test.txt')
#print 'Datos cargados'

clf = linear_model.LogisticRegression(C=1000000)
print 'Training'
clf.fit(X,y)

print 'Prediciendo'
ypred_train = clf.predict(X)
ypred_test = clf.predict(X_test)
print 'Calculando probabilidades'
prob = clf.predict_proba(X_test)

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
print 'Parametros del training set'
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

TP = 0
FP = 0
TN = 0
FN = 0
for i in range(len(y_test)):
    if y_test[i] == 1:
        if ypred_test[i] == 1:
            TP += 1
        else:
            FN += 1
    else:
        if ypred_test[i] == 1:
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
print 'Parametros del test set'
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

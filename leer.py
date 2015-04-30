# -*- coding: cp1252 -*-
import pickle as pk
'''Esto deberia servir para leer los datos de los .txt y guardarlos
   en diccionarios. Tambien para guardarlos con pickle y volver a
   abrirlos. Ha funcionado (tarda un poco...) con los datos del ar-
   chivo Split_1.txt.'''

i = 0

def leer(archivo = 'X.txt'):
    '''Lee los datos almacenados en ARCHIVO, que tienen que ser filas
       de datos separados por "|". Devuelve una tupla con dos diccio-
       narios, DICX y DICY, con los datos. Para cada fila, utiliza
       como clave para los diccionarios la primera palabra separada
       por "|" de la fila, y como valor la ultima palabra para DICY y
       una lista con el resto de palabras de la fila para DICX'''
    #Los campos 0, 3, 14, 16 y 41 no tenian numeros en todas las entradas de Split_1.txt.
    dicX = {}
    dicY = {}
    global i
    try:
        f = open(archivo,'r')
        for line in f:
            lista = line.split('|')[:-1] #Al final hay un '|' que hay que quitar
            #El primer campo, que estaba usando como clave del diccionario, da problemas. Pongo otra clave distinta (unica para cada transaccion, en teoria)
            dicX[i] = floating(lista[1:3] + lista[4:14] + lista[15:16] + lista[17:41] + lista[42:-1]) #el campo 14 daba problemas...
            dicY[i] = int(lista[-1])
            i += 1
        f.close()
        return dicX,dicY
    except:
        print 'Error al leer el archivo',archivo
        return None,None

def guardar(x,archivo = 'Xnew.txt'):
    '''Guarda la variable X en el archivo ARCHIVO con formato pickle'''
    try:
        f = open(archivo,'w')
        pk.dump(x,f)
        f.close()
    except:
        print 'Error al escribir en el archivo',archivo

def abrir(archivo = 'Xnew.txt'):
    '''Abre el archivo ARCHIVO con formato pickle y devuelve la variable
       contenida en este'''
    try:
        f = open(archivo,'r')
        var = pk.load(f)
        f.close()
        return var
    except:
        print 'Error al leer el archivo',archivo
        return None

def floating(lista):
    '''Recibe una lista de strings y devuelve una lista de enteros'''
    #Lo dejo con int o lo cambio a float??
    l = []
    global numero
    numero = 0
    for elemento in lista:
        l += [int(elemento)]
        numero += 1
    return l

##dicX,dicY = leer('training/Split_1.txt')
##guardar(dicX,'X_1.txt')
##guardar(dicY,'Y_1.txt')
##Xnew = abrir('X_1.txt')
##Ynew = abrir('Y_1.txt')
##print dicX == Xnew
##print dicY == Ynew



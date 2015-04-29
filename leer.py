import pickle as pk
'''Esto deberia servir para leer los datos de los .txt y guardarlos
   en diccionarios. Tambien para guardarlos con pickle y volver a
   abrirlos. De todas formas, con los datos de las transacciones
   mi ordenador peta al intentar guardarlos (Memory Error).
   Ha conseguido guardar los datos de Split_1.txt en diccionarios,
   pero guardarlos con pickle ha sido demasiado para el pobre...'''

def leer(archivo = 'X.txt'):
    '''Lee los datos almacenados en ARCHIVO, que tienen que ser filas
       de datos separados por "|". Devuelve una tupla con dos diccio-
       narios, DICX y DICY, con los datos. Para cada fila, utiliza
       como clave para los diccionarios la primera palabra separada
       por "|" de la fila, y como valor la ultima palabra para DICY y
       una lista con el resto de palabras de la fila para DICX'''
    dicX = {}
    dicY = {}
    try:
        f = open(archivo,'r')
        for line in f:
            lista = line.split('|')
            dicX[lista[0]] = lista[1:]
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

dicX,dicY = leer()
guardar(dicX)
guardar(dicY,'Ynew.txt')
Xnew = abrir()
Ynew = abrir('Ynew.txt')
print dicX == Xnew
print dicY == Ynew



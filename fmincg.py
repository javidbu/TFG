import numpy as np

def fmincg(f,fgrad,X,options=None):
    if options is None:
        pass
    else:
        if options.has_key('MaxIter'):
            length = options['MaxIter']
        else:
            length = 100
    RHO = 0.01
    SIG = 0.5
    INT = 0.1
    EXT = 3.0
    MAX = 20
    RATIO = 100
    red = 1
    S = 'Iteracion '

    i = 0
    ls_failed = 0
    fX = np.matrix(np.zeros((1,1))[0:0])
    f1 = f(X)
    df1 = fgrad(X)
    if length < 0: i += 1
    s = -df1
    d1 = np.float_(-s.transpose()*s)
    z1 = np.float_(red/(1-d1))

    while i < abs(length):
        if length > 0: i += 1
        (X0,f0,df0) = (X,f1,df1)
        X += np.float_(z1)*s
        f2 = f(X)
        df2 = fgrad(X)
        if length < 0: i += 1
        d2 = np.float_(df2.transpose()*s)
        (f3,d3,z3)=(f1,np.float_(d1),np.float_(-z1))
        if length > 0:
            M = MAX
        else: M = min(MAX,-length-i)
        (success,limit)=(0,-1) #AQUI
        while True:
            while f2 > (f1 + z1*RHO*d1) or d2 > (-SIG*d1) and M>0:
                limit = z1
                if f2>f1:
                    z2 = np.float_(z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3))
                else:
                    A = np.float_(6*(f2-f3)/z3+3*(d2+d3))
                    B = np.float_(3*(f3-f2)-z3*(d3+2*d2))
                    z2 = np.float_((np.sqrt(B*B-A*d2*z3*z3)-B)/A)
                if np.isnan(z2) or np.isinf(z2):
                    z2 = np.float_(z3/2)
                z2 = np.float_(max(min(z2, INT*z3),(1-INT)*z3))
                z1 += np.float_(z2)
                X += np.float_(z2)*s
                f2 = f(X)
                df2 = fgrad(X)
                M -= 1
                if length < 0: i += 1
                d2 = np.float_(df2.transpose()*s)
                z3 -= np.float_(z2)
            if f2 > f1+z1*RHO*d1 or d2 > -SIG*d1:
                break
            elif d2 > SIG*d1:
                success = 1
                break
            elif M == 0:
                break
            A = np.float_(6*(f2-f3)/z3+3*(d2+d3))
            B = np.float_(3*(f3-f2)-z3*(d3+2*d2))
            z2 = np.float_(-d2*z3*z3/(B+np.sqrt(B*B-A*d2*z3*z3)))
            if (not np.isreal(z2)) or np.isnan(z2) or np.isinf(z2) or z2 < 0:
                if limit < -0.5: z2 = np.float_(z1*(EXT-1))
                else: z2 = np.float_((limit-z1)/2.)
            elif limit > -0.5 and (z1+z2 > limit): z2 = np.float_((limit-z1)/2.)
            elif limit < -0.5 and (z1+z2 > z1*EXT): z2 = np.float_(z1*(EXT-1.0))
            elif z2 < -z3*INT: z2 = -z3*INT
            elif limit > -0.5 and z2 < (limit-z1)*(1.0-INT): z2 = np.float_((limit-z1)*(1.0-INT))
            (f3,d3,z3) = (f2,np.float_(d2),-z2)
            z1 += np.float_(z2)
            X += np.float_(z2)*s
            f2 = f(X)
            df2 = fgrad(X)
            M -= 1
            if length < 0: i += 1
            d2 = np.float_(df2.transpose()*s)

        if success != 0:
            f1 = f2
            fX = np.concatenate((fX.transpose(),np.matrix(f1)),axis = 1).transpose()
            print '%s %4i | Cost: %4.6e' % (S,i,f1)
            s = np.float_((df2.transpose()*df2-df1.transpose()*df2)/(df1.transpose()*df1))*s - df2
            (df1,df2)=(df2,df1)
            d2 = np.float_(df1.transpose()*s)
            if d2 > 0:
                s = -df1
                d2 = np.float_(-s.transpose()*s)
            z1 *= min(RATIO,np.float_(d1/(d2-np.finfo(np.double).tiny)))
            d1 = np.float_(d2)
            ls_failed = 0
        else:
            (X,f1,df1) = (X0,f0,df0)
            if ls_failed != 0 or i > abs(length):
                break
            (df1,df2)=(df2,df1)
            s = -df1
            d1 = np.float_(-s.transpose()*s)
            z1 = np.float_(1/(1-d1))
            ls_failed = 1
    return (X,fX,i)

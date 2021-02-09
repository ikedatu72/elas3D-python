# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:53:07 2020

@author: ki2844
"""

import numpy as np
#  Subroutine that carries out the conjugate gradient relaxation process

def dembx(ns, Lstep, gg, dk, gtest, ldemb, kkk, gb, ib, pix, u, h):
    #  Initialize the conjugate direction vector on first call to dembx only
    #  For calls to dembx after the first, we want to continue using the 
    #  value of h determined in the previous call. Of course, if npoints is
    #  greater than 1, this initialization step will be run for every new
    #  microstructure used, as kkk is reset to 1 every time the counter micro
    #  is increased. 
    
    if kkk == 0:
        h = gb.copy()
    
    #  Lstep counts the number of conjugate gradient steps taken in
    #  each call to dembx
    Lstep = 0
    
    for ijk in range(ldemb):
        Lstep = Lstep + 1 
        
        Ah = np.zeros((ns, 3))
        
        #  Do global matrix multiply via small stiffness matrices, Ah = A * h
        #  The long statement below correctly brings in all the terms from 
        #  the global matrix A using only the small stiffness matrices dk.
        for j in range(3):
            for n in range(3):
                for m in range(ns):
                    Ah[m,j]=Ah[m,j]+h[ib[m,0],n]*( dk[pix[ib[m,26]],0,j,3,n]\
                    +dk[pix[ib[m,6]],1,j,2,n]\
                    +dk[pix[ib[m,24]],4,j,7,n]+dk[pix[ib[m,14]],5,j,6,n] )+\
                    h[ib[m,1],n]*( dk[pix[ib[m,26]],0,j,2,n]\
                    +dk[pix[ib[m,24]],4,j,6,n] )+\
                    h[ib[m,2],n]*( dk[pix[ib[m,26]],0,j,1,n]+dk[pix[ib[m,4]],3,j,2,n]+\
                    dk[pix[ib[m,12]],7,j,6,n]+dk[pix[ib[m,24]],4,j,5,n] )+\
                    h[ib[m,3],n]*( dk[pix[ib[m,4]],3,j,1,n]\
                    +dk[pix[ib[m,12]],7,j,5,n] )+\
                    h[ib[m,4],n]*( dk[pix[ib[m,5]],2,j,1,n]+dk[pix[ib[m,4]],3,j,0,n]+\
                    dk[pix[ib[m,13]],6,j,5,n]+dk[pix[ib[m,12]],7,j,4,n] )+\
                    h[ib[m,5],n]*( dk[pix[ib[m,5]],2,j,0,n]\
                    +dk[pix[ib[m,13]],6,j,4,n] )+\
                    h[ib[m,6],n]*( dk[pix[ib[m,5]],2,j,3,n]+dk[pix[ib[m,6]],1,j,0,n]+\
                    dk[pix[ib[m,13]],6,j,7,n]+dk[pix[ib[m,14]],5,j,4,n] )+\
                    h[ib[m,7],n]*( dk[pix[ib[m,6]],1,j,3,n]\
                    +dk[pix[ib[m,14]],5,j,7,n] )+\
                    h[ib[m,8],n]*( dk[pix[ib[m,24]],4,j,3,n]\
                    +dk[pix[ib[m,14]],5,j,2,n] )+\
                    h[ib[m,9],n]*( dk[pix[ib[m,24]],4,j,2,n] )+\
                    h[ib[m,10],n]*( dk[pix[ib[m,12]],7,j,2,n]\
                    +dk[pix[ib[m,24]],4,j,1,n] )+\
                    h[ib[m,11],n]*( dk[pix[ib[m,12]],7,j,1,n] )+\
                    h[ib[m,12],n]*( dk[pix[ib[m,12]],7,j,0,n]\
                    +dk[pix[ib[m,13]],6,j,1,n] )+\
                    h[ib[m,13],n]*( dk[pix[ib[m,13]],6,j,0,n] )+\
                    h[ib[m,14],n]*( dk[pix[ib[m,13]],6,j,3,n]\
                    +dk[pix[ib[m,14]],5,j,0,n] )+\
                    h[ib[m,15],n]*( dk[pix[ib[m,14]],5,j,3,n] )+\
                    h[ib[m,16],n]*( dk[pix[ib[m,26]],0,j,7,n]\
                    +dk[pix[ib[m,6]],1,j,6,n] )+\
                    h[ib[m,17],n]*( dk[pix[ib[m,26]],0,j,6,n] )+\
                    h[ib[m,18],n]*( dk[pix[ib[m,26]],0,j,5,n]\
                    +dk[pix[ib[m,4]],3,j,6,n] )+\
                    h[ib[m,19],n]*( dk[pix[ib[m,4]],3,j,5,n] )+\
                    h[ib[m,20],n]*( dk[pix[ib[m,4]],3,j,4,n]\
                    +dk[pix[ib[m,5]],2,j,5,n] )+\
                    h[ib[m,21],n]*( dk[pix[ib[m,5]],2,j,4,n] )+\
                    h[ib[m,22],n]*( dk[pix[ib[m,5]],2,j,7,n]\
                    +dk[pix[ib[m,6]],1,j,4,n] )+\
                    h[ib[m,23],n]*( dk[pix[ib[m,6]],1,j,7,n] )+\
                    h[ib[m,24],n]*( dk[pix[ib[m,13]],6,j,2,n]\
                    +dk[pix[ib[m,12]],7,j,3,n]+\
                    dk[pix[ib[m,14]],5,j,1,n]+dk[pix[ib[m,24]],4,j,0,n] )+\
                    h[ib[m,25],n]*( dk[pix[ib[m,5]],2,j,6,n]\
                    +dk[pix[ib[m,4]],3,j,7,n]+\
                    dk[pix[ib[m,26]],0,j,4,n]+dk[pix[ib[m,6]],1,j,5,n] )+\
                    h[ib[m,26],n]*( dk[pix[ib[m,26]],0,j,0,n]\
                    +dk[pix[ib[m,6]],1,j,1,n]+\
                    dk[pix[ib[m,5]],2,j,2,n]+dk[pix[ib[m,4]],3,j,3,n]\
                    +dk[pix[ib[m,24]],4,j,4,n]+\
                    dk[pix[ib[m,14]],5,j,5,n]+dk[pix[ib[m,13]],6,j,6,n]+\
                    dk[pix[ib[m,12]],7,j,7,n] )
                        
        hAh = 0.0
        for m3 in range(3):
            for m in range(ns):
                hAh = hAh + h[m, m3] * Ah[m, m3]
                
        lambda0 = gg/hAh 
        
        for m3 in range(3):
            for m in range(ns):
                u[m, m3] = u[m, m3]- lambda0*h[m, m3]
                gb[m, m3] = gb[m, m3] - lambda0*Ah[m, m3]
                
        gglast = gg
        
        gg = 0.0
        
        for m3 in range(3):
            for m in range(ns):
                gg = gg + gb[m, m3]*gb[m, m3]
                
        if gg < gtest: 
            break
        gamma = gg/gglast
        for m3 in range(3):
            for m in range(ns):
                h[m, m3] = gb[m, m3] + gamma*h[m, m3]
                
    return Ah, Lstep, h, u, gg

if __name__ == "__main__":
    pass

    """
    ns = 6000
    Lstep = 0
    gg = 25.832790798609736
    dk = np.random.randn(2, 8, 3, 8, 3) 
    gtest = 6e-5 
    ldemb = 50
    kkk = 0 
    gb = np.random.randn(6000, 3) 
    ib = np.random.randn(6000, 27)
    pix = np.random.randint(0, 2, (6000))
    u = np.random.randn(6000, 3) 
    h = 0
    
    Ah, Lstep, h, u, gg = dembx(ns, Lstep, gg, dk, gtest, ldemb, kkk, gb, ib, pix, u, h)
    """
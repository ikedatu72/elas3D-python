# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:53:07 2020

@author: ki2844
"""

import numpy as np
from numba import jit
#  Subroutine that carries out the conjugate gradient relaxation process

@jit(nopython = True)
def dembx(ns, Lstep, gg, dk, gtest, ldemb, kkk, gb, ib, pix, u, h):
    #  Initialize the conjugate direction vector on first call to dembx only
    #  For calls to dembx after the first, we want to continue using the 
    #  value of h determined in the previous call. Of course, if npoints is
    #  greater than 1, this initialization step will be run for every new
    #  microstructure used, as kkk is reset to 1 every time the counter micro
    #  is increased. 
    
    #if kkk == 0:
    #    h = gb.copy()
    
    #  Lstep counts the number of conjugate gradient steps taken in
    #  each call to dembx
    Lstep = 0
    
    hAh = 0.0
    for _ in range(ldemb):
        Lstep = Lstep + 1
    
        Ah = np.zeros((ns, 3), dtype=np.float_)
        
        #  Do global matrix multiply via small stiffness matrices, Ah = A * h
        #  The long statement below correctly brings in all the terms from 
        #  the global matrix A using only the small stiffness matrices dk.
        
        for j in range(3):
            for n in range(3):
                Ah[:,j] = Ah[:,j] + h[ib[:,0],n]*( dk[pix[ib[:,26]],0,j,3,n]\
                +dk[pix[ib[:,6]],1,j,2,n]\
                +dk[pix[ib[:,24]],4,j,7,n]+dk[pix[ib[:,14]],5,j,6,n] )+\
                h[ib[:,1],n]*( dk[pix[ib[:,26]],0,j,2,n]\
                +dk[pix[ib[:,24]],4,j,6,n] )+\
                h[ib[:,2],n]*( dk[pix[ib[:,26]],0,j,1,n]+dk[pix[ib[:,4]],3,j,2,n]+\
                dk[pix[ib[:,12]],7,j,6,n]+dk[pix[ib[:,24]],4,j,5,n] )+\
                h[ib[:,3],n]*( dk[pix[ib[:,4]],3,j,1,n]\
                +dk[pix[ib[:,12]],7,j,5,n] )+\
                h[ib[:,4],n]*( dk[pix[ib[:,5]],2,j,1,n]+dk[pix[ib[:,4]],3,j,0,n]+\
                dk[pix[ib[:,13]],6,j,5,n]+dk[pix[ib[:,12]],7,j,4,n] )+\
                h[ib[:,5],n]*( dk[pix[ib[:,5]],2,j,0,n]\
                +dk[pix[ib[:,13]],6,j,4,n] )+\
                h[ib[:,6],n]*( dk[pix[ib[:,5]],2,j,3,n]+dk[pix[ib[:,6]],1,j,0,n]+\
                dk[pix[ib[:,13]],6,j,7,n]+dk[pix[ib[:,14]],5,j,4,n] )+\
                h[ib[:,7],n]*( dk[pix[ib[:,6]],1,j,3,n]\
                +dk[pix[ib[:,14]],5,j,7,n] )+\
                h[ib[:,8],n]*( dk[pix[ib[:,24]],4,j,3,n]\
                +dk[pix[ib[:,14]],5,j,2,n] )+\
                h[ib[:,9],n]*( dk[pix[ib[:,24]],4,j,2,n] )+\
                h[ib[:,10],n]*( dk[pix[ib[:,12]],7,j,2,n]\
                +dk[pix[ib[:,24]],4,j,1,n] )+\
                h[ib[:,11],n]*( dk[pix[ib[:,12]],7,j,1,n] )+\
                h[ib[:,12],n]*( dk[pix[ib[:,12]],7,j,0,n]\
                +dk[pix[ib[:,13]],6,j,1,n] )+\
                h[ib[:,13],n]*( dk[pix[ib[:,13]],6,j,0,n] )+\
                h[ib[:,14],n]*( dk[pix[ib[:,13]],6,j,3,n]\
                +dk[pix[ib[:,14]],5,j,0,n] )+\
                h[ib[:,15],n]*( dk[pix[ib[:,14]],5,j,3,n] )+\
                h[ib[:,16],n]*( dk[pix[ib[:,26]],0,j,7,n]\
                +dk[pix[ib[:,6]],1,j,6,n] )+\
                h[ib[:,17],n]*( dk[pix[ib[:,26]],0,j,6,n] )+\
                h[ib[:,18],n]*( dk[pix[ib[:,26]],0,j,5,n]\
                +dk[pix[ib[:,4]],3,j,6,n] )+\
                h[ib[:,19],n]*( dk[pix[ib[:,4]],3,j,5,n] )+\
                h[ib[:,20],n]*( dk[pix[ib[:,4]],3,j,4,n]\
                +dk[pix[ib[:,5]],2,j,5,n] )+\
                h[ib[:,21],n]*( dk[pix[ib[:,5]],2,j,4,n] )+\
                h[ib[:,22],n]*( dk[pix[ib[:,5]],2,j,7,n]\
                +dk[pix[ib[:,6]],1,j,4,n] )+\
                h[ib[:,23],n]*( dk[pix[ib[:,6]],1,j,7,n] )+\
                h[ib[:,24],n]*( dk[pix[ib[:,13]],6,j,2,n]\
                +dk[pix[ib[:,12]],7,j,3,n]+\
                dk[pix[ib[:,14]],5,j,1,n]+dk[pix[ib[:,24]],4,j,0,n] )+\
                h[ib[:,25],n]*( dk[pix[ib[:,5]],2,j,6,n]\
                +dk[pix[ib[:,4]],3,j,7,n]+\
                dk[pix[ib[:,26]],0,j,4,n]+dk[pix[ib[:,6]],1,j,5,n] )+\
                h[ib[:,26],n]*( dk[pix[ib[:,26]],0,j,0,n]\
                +dk[pix[ib[:,6]],1,j,1,n]+\
                dk[pix[ib[:,5]],2,j,2,n]+dk[pix[ib[:,4]],3,j,3,n]\
                +dk[pix[ib[:,24]],4,j,4,n]+\
                dk[pix[ib[:,14]],5,j,5,n]+dk[pix[ib[:,13]],6,j,6,n]+\
                dk[pix[ib[:,12]],7,j,7,n] )
                
        u = u - gg/np.sum(h*Ah)*h
        gb = gb - gg/np.sum(h*Ah)*Ah
    
        gglast = gg
        gg = np.sum((gb**2))
        
        if gg < gtest: 
            break
        gamma = gg/gglast

        h = gb + gamma*h
        
    return Ah, Lstep, h, u, gg

if __name__ == "__main__":
    pass

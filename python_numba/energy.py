# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 17:33:42 2020

@author: Ken Ikeda
@email: ikeda.ken@utexas.edu

Ph.D. candidate 
Jackson School of Geosciences
The University of Texas at Austin, Texas, USA
"""

import numpy as np
from numba import jit

#  Subroutine computes the total energy, utot, and the gradient, gb
@jit(nopython = True)
def energy(ns, C, ib, u, pix, dk, b):
    
    gb = np.zeros((ns, 3), dtype=np.float_)
    
    #  Do global matrix multiply via small stiffness matrices, gb = A * u
    #  The long statement below correctly brings in all the terms from 
    #  the global matrix A using only the small stiffness matrices.
    
    for j in range(3):
        for n in range(3):
            gb[:,j] = gb[:,j]+u[ib[:,0],n]*( dk[pix[ib[:,26]],0,j,3,n]\
                        +dk[pix[ib[:,6]],1,j,2,n]\
                        +dk[pix[ib[:,24]],4,j,7,n]+dk[pix[ib[:,14]],5,j,6,n] )+\
                        u[ib[:,1],n]*( dk[pix[ib[:,26]],0,j,2,n]\
                        +dk[pix[ib[:,24]],4,j,6,n] )+\
                        u[ib[:,2],n]*( dk[pix[ib[:,26]],0,j,1,n]+dk[pix[ib[:,4]],3,j,2,n]+\
                        dk[pix[ib[:,12]],7,j,6,n]+dk[pix[ib[:,24]],4,j,5,n] )+\
                        u[ib[:,3],n]*( dk[pix[ib[:,4]],3,j,1,n]\
                        +dk[pix[ib[:,12]],7,j,5,n] )+\
                        u[ib[:,4],n]*( dk[pix[ib[:,5]],2,j,1,n]+dk[pix[ib[:,4]],3,j,0,n]+\
                        dk[pix[ib[:,13]],6,j,5,n]+dk[pix[ib[:,12]],7,j,4,n] )+\
                        u[ib[:,5],n]*( dk[pix[ib[:,5]],2,j,0,n]\
                        +dk[pix[ib[:,13]],6,j,4,n] )+\
                        u[ib[:,6],n]*( dk[pix[ib[:,5]],2,j,3,n]+dk[pix[ib[:,6]],1,j,0,n]+\
                        dk[pix[ib[:,13]],6,j,7,n]+dk[pix[ib[:,14]],5,j,4,n] )+\
                        u[ib[:,7],n]*( dk[pix[ib[:,6]],1,j,3,n]\
                        +dk[pix[ib[:,14]],5,j,7,n] )+\
                        u[ib[:,8],n]*( dk[pix[ib[:,24]],4,j,3,n]\
                        +dk[pix[ib[:,14]],5,j,2,n] )+\
                        u[ib[:,9],n]*( dk[pix[ib[:,24]],4,j,2,n] )+\
                        u[ib[:,10],n]*( dk[pix[ib[:,12]],7,j,2,n]\
                        +dk[pix[ib[:,24]],4,j,1,n] )+\
                        u[ib[:,11],n]*( dk[pix[ib[:,12]],7,j,1,n] )+\
                        u[ib[:,12],n]*( dk[pix[ib[:,12]],7,j,0,n]\
                        +dk[pix[ib[:,13]],6,j,1,n] )+\
                        u[ib[:,13],n]*( dk[pix[ib[:,13]],6,j,0,n] )+\
                        u[ib[:,14],n]*( dk[pix[ib[:,13]],6,j,3,n]\
                        +dk[pix[ib[:,14]],5,j,0,n] )+\
                        u[ib[:,15],n]*( dk[pix[ib[:,14]],5,j,3,n] )+\
                        u[ib[:,16],n]*( dk[pix[ib[:,26]],0,j,7,n]\
                        +dk[pix[ib[:,6]],1,j,6,n] )+\
                        u[ib[:,17],n]*( dk[pix[ib[:,26]],0,j,6,n] )+\
                        u[ib[:,18],n]*( dk[pix[ib[:,26]],0,j,5,n]\
                        +dk[pix[ib[:,4]],3,j,6,n] )+\
                        u[ib[:,19],n]*( dk[pix[ib[:,4]],3,j,5,n] )+\
                        u[ib[:,20],n]*( dk[pix[ib[:,4]],3,j,4,n]\
                        +dk[pix[ib[:,5]],2,j,5,n] )+\
                        u[ib[:,21],n]*( dk[pix[ib[:,5]],2,j,4,n] )+\
                        u[ib[:,22],n]*( dk[pix[ib[:,5]],2,j,7,n]\
                        +dk[pix[ib[:,6]],1,j,4,n] )+\
                        u[ib[:,23],n]*( dk[pix[ib[:,6]],1,j,7,n] )+\
                        u[ib[:,24],n]*( dk[pix[ib[:,13]],6,j,2,n]\
                        +dk[pix[ib[:,12]],7,j,3,n]+\
                        dk[pix[ib[:,14]],5,j,1,n]+dk[pix[ib[:,24]],4,j,0,n] )+\
                        u[ib[:,25],n]*( dk[pix[ib[:,5]],2,j,6,n]\
                        +dk[pix[ib[:,4]],3,j,7,n]+\
                        dk[pix[ib[:,26]],0,j,4,n]+dk[pix[ib[:,6]],1,j,5,n] )+\
                        u[ib[:,26],n]*( dk[pix[ib[:,26]],0,j,0,n]\
                        +dk[pix[ib[:,6]],1,j,1,n]+\
                        dk[pix[ib[:,5]],2,j,2,n]+dk[pix[ib[:,4]],3,j,3,n]\
                        +dk[pix[ib[:,24]],4,j,4,n]+\
                        dk[pix[ib[:,14]],5,j,5,n]+dk[pix[ib[:,13]],6,j,6,n]+\
                        dk[pix[ib[:,12]],7,j,7,n] )

    utot = C + 0.5*np.sum(u*gb) + np.sum(b*u)
    gb = gb + b
    
    return gb, utot
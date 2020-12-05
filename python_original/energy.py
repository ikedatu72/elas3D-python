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

#  Subroutine computes the total energy, utot, and the gradient, gb
def energy(ns, C, ib, u, pix, dk, b):
    gb = np.zeros((ns, 3))
    
    #  Do global matrix multiply via small stiffness matrices, gb = A * u
    #  The long statement below correctly brings in all the terms from 
    #  the global matrix A using only the small stiffness matrices.
    
    for j in range(3):
        for n in range(3):
            for m in range(ns):
                gb[m,j] = gb[m,j]+u[ib[m,0],n]*( dk[pix[ib[m,26]],0,j,3,n]\
                            +dk[pix[ib[m,6]],1,j,2,n]\
                            +dk[pix[ib[m,24]],4,j,7,n]+dk[pix[ib[m,14]],5,j,6,n] )+\
                            u[ib[m,1],n]*( dk[pix[ib[m,26]],0,j,2,n]\
                            +dk[pix[ib[m,24]],4,j,6,n] )+\
                            u[ib[m,2],n]*( dk[pix[ib[m,26]],0,j,1,n]+dk[pix[ib[m,4]],3,j,2,n]+\
                            dk[pix[ib[m,12]],7,j,6,n]+dk[pix[ib[m,24]],4,j,5,n] )+\
                            u[ib[m,3],n]*( dk[pix[ib[m,4]],3,j,1,n]\
                            +dk[pix[ib[m,12]],7,j,5,n] )+\
                            u[ib[m,4],n]*( dk[pix[ib[m,5]],2,j,1,n]+dk[pix[ib[m,4]],3,j,0,n]+\
                            dk[pix[ib[m,13]],6,j,5,n]+dk[pix[ib[m,12]],7,j,4,n] )+\
                            u[ib[m,5],n]*( dk[pix[ib[m,5]],2,j,0,n]\
                            +dk[pix[ib[m,13]],6,j,4,n] )+\
                            u[ib[m,6],n]*( dk[pix[ib[m,5]],2,j,3,n]+dk[pix[ib[m,6]],1,j,0,n]+\
                            dk[pix[ib[m,13]],6,j,7,n]+dk[pix[ib[m,14]],5,j,4,n] )+\
                            u[ib[m,7],n]*( dk[pix[ib[m,6]],1,j,3,n]\
                            +dk[pix[ib[m,14]],5,j,7,n] )+\
                            u[ib[m,8],n]*( dk[pix[ib[m,24]],4,j,3,n]\
                            +dk[pix[ib[m,14]],5,j,2,n] )+\
                            u[ib[m,9],n]*( dk[pix[ib[m,24]],4,j,2,n] )+\
                            u[ib[m,10],n]*( dk[pix[ib[m,12]],7,j,2,n]\
                            +dk[pix[ib[m,24]],4,j,1,n] )+\
                            u[ib[m,11],n]*( dk[pix[ib[m,12]],7,j,1,n] )+\
                            u[ib[m,12],n]*( dk[pix[ib[m,12]],7,j,0,n]\
                            +dk[pix[ib[m,13]],6,j,1,n] )+\
                            u[ib[m,13],n]*( dk[pix[ib[m,13]],6,j,0,n] )+\
                            u[ib[m,14],n]*( dk[pix[ib[m,13]],6,j,3,n]\
                            +dk[pix[ib[m,14]],5,j,0,n] )+\
                            u[ib[m,15],n]*( dk[pix[ib[m,14]],5,j,3,n] )+\
                            u[ib[m,16],n]*( dk[pix[ib[m,26]],0,j,7,n]\
                            +dk[pix[ib[m,6]],1,j,6,n] )+\
                            u[ib[m,17],n]*( dk[pix[ib[m,26]],0,j,6,n] )+\
                            u[ib[m,18],n]*( dk[pix[ib[m,26]],0,j,5,n]\
                            +dk[pix[ib[m,4]],3,j,6,n] )+\
                            u[ib[m,19],n]*( dk[pix[ib[m,4]],3,j,5,n] )+\
                            u[ib[m,20],n]*( dk[pix[ib[m,4]],3,j,4,n]\
                            +dk[pix[ib[m,5]],2,j,5,n] )+\
                            u[ib[m,21],n]*( dk[pix[ib[m,5]],2,j,4,n] )+\
                            u[ib[m,22],n]*( dk[pix[ib[m,5]],2,j,7,n]\
                            +dk[pix[ib[m,6]],1,j,4,n] )+\
                            u[ib[m,23],n]*( dk[pix[ib[m,6]],1,j,7,n] )+\
                            u[ib[m,24],n]*( dk[pix[ib[m,13]],6,j,2,n]\
                            +dk[pix[ib[m,12]],7,j,3,n]+\
                            dk[pix[ib[m,14]],5,j,1,n]+dk[pix[ib[m,24]],4,j,0,n] )+\
                            u[ib[m,25],n]*( dk[pix[ib[m,5]],2,j,6,n]\
                            +dk[pix[ib[m,4]],3,j,7,n]+\
                            dk[pix[ib[m,26]],0,j,4,n]+dk[pix[ib[m,6]],1,j,5,n] )+\
                            u[ib[m,26],n]*( dk[pix[ib[m,26]],0,j,0,n]\
                            +dk[pix[ib[m,6]],1,j,1,n]+\
                            dk[pix[ib[m,5]],2,j,2,n]+dk[pix[ib[m,4]],3,j,3,n]\
                            +dk[pix[ib[m,24]],4,j,4,n]+\
                            dk[pix[ib[m,14]],5,j,5,n]+dk[pix[ib[m,13]],6,j,6,n]+\
                            dk[pix[ib[m,12]],7,j,7,n] )
    utot = C
    for m3 in range(3):
        for m in range(ns):
            utot = utot + 0.5*u[m,m3]*gb[m,m3] + b[m,m3]*u[m,m3]
            gb[m, m3] = gb[m, m3] + b[m, m3]
            
    return gb, utot
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:27:47 2020

@author: Ken Ikeda
@email: ikeda.ken@utexas.edu

Ph.D. candidate 
Jackson School of Geosciences
The University of Texas at Austin, Texas, USA
"""

import numpy as np

#  Subroutine that sets up microstructural image
def ppixel(nx, ny, nz, ns, nphase):
    pix = np.zeros(ns)

    #  (USER)  If you want to set up a test image inside the program, instead of
    #  reading it in from a file, this should be done inside this subroutine.
    
    f = open("microstructure.dat", "r")

    nxy = nx*ny
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                m = nxy*(k) + nx*(j) + i
                pix[m] = int(f.readline())
    
    #  Check for wrong phase labels--less than 1 or greater than nphase
    #  FOR PYTHON, label starts from 0. So we check for any label which is less than 0
    
    for m in range(ns):
        if pix[m] < 0:
            print("Phase label in pix < 0--error at {}".format(m))
        if pix[m] > nphase-1:
            print("Phase label in pix > nphase--error at {}".format(m))
            
    return pix.astype('int')

if __name__ == "__main__":
    pix = ppixel(10, 20, 30, 10*20*30, 2)                 
    
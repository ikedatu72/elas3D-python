# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 16:19:52 2020

@author: Ken Ikeda
@email: ikeda.ken@utexas.edu

Ph.D. candidate 
Jackson School of Geosciences
The University of Texas at Austin, Texas, USA
"""

import numpy as np

#  Subroutine that counts volume fractions
def assig(ns, nphase, pix):
    #Note --> Fortran subroutine takes (ns, nphase, prob) as an input 
    
    prob = np.zeros(nphase)
    
    for m in range(ns):
        for i in range(nphase):
            if pix[m] == i:
                prob[i] = prob[i] + 1
    
    for i in range(nphase):
        prob[i] = prob[i]/ns

    return prob

if __name__ == "__main__":
    pix = np.random.randint(low = 0, high = 2, size = (10, 20, 30)).flatten(order = 'F')
    nphase = 2
    ns = 10*20*30
    
    prob = assig(ns, nphase, pix)
    print(prob)
    
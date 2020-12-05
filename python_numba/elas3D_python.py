# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 23:03:35 2020

@author: Ken Ikeda
@email: ikeda.ken@utexas.edu

Ph.D. candidate 
Jackson School of Geosciences
The University of Texas at Austin, Texas, USA
"""

import numpy as np

from ppixel import ppixel 
from assig import assig
from femat import femat
from energy import energy
from dembx import dembx
from stress import stress

print("#####################################")
print("####Starting elas3D python version###")
print("#####################################")

##########################################
## The main code goes below this line ####
# %% Cube inspection 
ny = 10
nx = 20
nz = 30

ns = nx*ny*nz
gtest = 1.e-8*ns
nphase = 2
print("nx = {}, ny = {}, nz = {}".format(nx, ny, nz))

#constructing phasemod 
phasemod = np.zeros((nphase, 2)) 
phasemod[0, 0] = 1.0
phasemod[0, 1] = 0.2
phasemod[1, 0] = 0.5
phasemod[1, 1] = 0.2

#changing phasemod from Young's modulus + Possion ratio to Bulk and Shear modulus
for i in range(nphase):
    E = phasemod[i, 0]
    phasemod[i, 0] = phasemod[i, 0]/(3 * (1 - 2 * phasemod[i, 1]))
    phasemod[i, 1] = E/(2 * (1 + phasemod[i, 1]))
    
# construct the neighbor table, ib(m,n) 
# First construct the 27 neighbor table in terms of delta i, delta j, and
# delta k information (see Table 3 in manual)

in0 = np.zeros(27)   #Frotran uses "in" as a variable name, but "in" is an invalid variable name in python
jn0 = np.zeros(27)
kn0 = np.zeros(27)

in0[0] = 0;     in0[1:4] = 1;   in0[4] = 0;     in0[5:8] = -1;
jn0[0:2] = 1;   jn0[2] = 0;     jn0[3:6] = -1;  jn0[6] = 0;     jn0[7] = 1;

for i in range(8):
    kn0[i] = 0
    kn0[i+8] = -1
    kn0[i+16] = 1
    in0[i+8] = in0[i]
    in0[i+16] = in0[i]
    jn0[i+8] = jn0[i]
    jn0[i+16] = jn0[i]    

in0[24:27] = 0
jn0[24:27] = 0
kn0[24] = -1
kn0[25] = 1
kn0[26]= 0

#  Now construct neighbor table according to 1-d labels
#  Matrix ib(m,n) gives the 1-d label of the n'th neighbor (n=1,27) of
#  the node labelled m.
ib = np.zeros((ns, 27), dtype = 'int')
nxy = nx*ny
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            m = nxy*(k) + nx*(j) + i
            for n in range(27):
                i1 = i + in0[n]
                j1 = j + jn0[n]
                k1 = k + kn0[n]
                if i1 < 0:
                    i1 = i1 + nx
                if i1 > nx-1:
                    i1 = i1 - nx
                if j1 < 0:
                    j1 = j1 + ny
                if j1 > ny-1:
                    j1 = j1 - ny
                if k1 < 0:
                    k1 = k1 + nz
                if k1 > nz-1: 
                    k1 = k1 - nz
                m1 = nxy*(k1) + nx*(j1) + i1
                ib[m,n] = m1
                
# Compute the average stress and strain in each microstructure.       
# (USER) npoints is the number of microstructures to use.

npoints = 1

for micro in range(npoints):
    #  Read in a microstructure in subroutine ppixel, and set up pix(m)
    #  with the appropriate phase assignments.
    pix = ppixel(nx, ny, nz, ns, nphase)

    #  Count and output the volume fractions of the different phases
    prob = assig(ns, nphase, pix)
    
    for i in range(nphase):
        print("Phase {}, bulk = {:12.6f}, shear = {:12.6f}".format(i, phasemod[i, 0], phasemod[i, 1]))
        
    for i in range(nphase):
        print("Volume fraction of phase {} is {:8.5f}".format(i, prob[i]))
    
    #  (USER) Set applied strains
    #  Actual shear strain applied in do 1050 loop is exy, exz, and eyz as
    #  given in the statements below.  The engineering shear strain, by which
    #  the shear modulus is usually defined, is twice these values.
    
    exx = 0.1
    eyy = 0.1
    ezz = 0.1
    exz = 0.1/2
    eyz = 0.2/2
    exy = 0.3/2
    
    print("Applied engineering strains: exx eyy ezz exz eyz exy")
    print("{} {} {} {} {} {}".format(exx, eyy, ezz, 2*exz, 2*eyz, 2*exy))
    
    
    # Set up the elastic modulus variables, finite element stiffness matrices,
    # the constant, C, and vector, b, required for computing the energy.
    # (USER) If anisotropic elastic moduli tensors are used, these need to be
    # input in subroutine femat.
    
    C,dk,b,cmod = femat(nx,ny,nz,ns,exx,eyy,ezz,exz,eyz,exy,phasemod,nphase,pix,ib)
    
    # Apply chosen strains as a homogeneous macroscopic strain 
    # as the initial condition.
    
    u = np.zeros((ns, 3))
    for k in range(1, nz+1):
        for j in range(1, ny+1):
            for i in range(1, nx+1):
                m = nxy*(k-1) + nx*(j-1) + i - 1
                x = i - 1
                y = j - 1
                z = k - 1
                u[m,0]=x*exx+y*exy+z*exz
                u[m,1]=x*exy+y*eyy+z*eyz
                u[m,2]=x*exz+y*eyz+z*ezz
                
    #  RELAXATION LOOP
    #  (USER) kmax is the maximum number of times dembx will be called, with
    #  ldemb conjugate gradient steps performed during each call.  The total
    #  number of conjugate gradient steps allowed for a given elastic
    #  computation is kmax*ldemb.
    
    kmax = 40
    ldemb = 50
    ltot = 0
    Lstep = 0
    h = 0   #initialization
    
    #  Call energy to get initial energy and initial gradient
    gb, utot = energy(ns, C, ib, u, pix, dk, b)
    
    #  gg is the norm squared of the gradient (gg=gb*gb)    
    gg = np.sum((gb**2))
            
    print("Initial energy = {:.4f} gg = {:.4f}".format(utot, gg))

    for kkk in range(kmax):
        #  call dembx to go into the conjugate gradient solver
        if kkk == 0:
            Ah, Lstep, h, u, gg = dembx(ns, Lstep, gg, dk, gtest, ldemb, kkk, gb, ib, pix, u, gb.copy())
        else:
            Ah, Lstep, h, u, gg = dembx(ns, Lstep, gg, dk, gtest, ldemb, kkk, gb, ib, pix, u, h)
            
        ltot = ltot + Lstep
        
        #  Call energy to compute energy after dembx call. If gg < gtest, this
        #  will be the final energy.  If gg is still larger than gtest, then this
        #  will give an intermediate energy with which to check how the 
        #  relaxation process is coming along.
        gb, utot = energy(ns, C, ib, u, pix, dk, b)
        
        print("Energy = {:.4f} gg = {:.4f}".format(utot, gg))
        print("Number of conjugate steps = {}".format(ltot))
        
        #  If relaxation process is finished, jump out of loop
        if gg < gtest:
            break
        
        #  If relaxation process will continue, compute and output stresses
        #  and strains as an additional aid to judge how the 
        #  relaxation procedure is progressing.
        
        strxx, stryy, strzz, strxz, stryz, strxy, sxx, syy, szz, sxz, syz, sxy = stress(nx, ny, nz, ns, exx, eyy, exz, eyz, ezz, exy, u, ib, pix, cmod)
        
        print("stresses: xx, yy, zz, xz, yz, xy: {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}".format(strxx, stryy, strzz, strxz, stryz, strxy))
        print("strains: xx, yy, zz, xz, yz, xy: {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}".format(sxx, syy, szz, sxz, syz, sxy))

    #   (Writer comment) The conjugate gradient iteration stops, final calculation of stress and strain 
    strxx, stryy, strzz, strxz, stryz, strxy, sxx, syy, szz, sxz, syz, sxy = stress(nx, ny, nz, ns, exx, eyy, exz, eyz, ezz, exy, u, ib, pix, cmod)
    
    print("stresses: xx, yy, zz, xz, yz, xy: {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}".format(strxx, stryy, strzz, strxz, stryz, strxy))
    print("strains: xx, yy, zz, xz, yz, xy: {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}".format(sxx, syy, szz, sxz, syz, sxy))
    

print("#####################################")
print("####Ending elas3D python version#####")
print("#####################################")
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
#  Subroutine that sets up the elastic moduli variables, 
#  the stiffness matrices,dk, the linear term in
#  displacements, b, and the constant term, C, that appear in the total energy 
#  due to the periodic boundary conditions

@jit(nopython = True)
def femat(nx,ny,nz,ns,exx,eyy,ezz,exz,eyz,exy,phasemod,nphase,pix,ib):
    nxy = nx*ny
    
    #  (USER) NOTE:  complete elastic modulus matrix is used, so an anisotropic
    #  matrix could be directly input at any point, since program is written 
    #  to use a general elastic moduli tensor, but is only explicitly 
    #  implemented for isotropic materials.
    
    #  initialize stiffness matrices
    dk = np.zeros((nphase, 8, 3, 8, 3), dtype=np.float_)
    
    # set up elastic moduli matrices for each kind of element
    #  ck and cmu are the bulk and shear modulus matrices, which need to be
    #  weighted by the actual bulk and shear moduli
    ck = np.zeros((6, 6), dtype=np.float_)
    cmu = np.zeros((6, 6), dtype=np.float_)
    
    ck[0, 0]=1.0
    ck[0, 1]=1.0
    ck[0, 2]=1.0
    ck[0, 3]=0.0
    ck[0, 4]=0.0
    ck[0, 5]=0.0
    ck[1, 0]=1.0
    ck[1, 1]=1.0
    ck[1, 2]=1.0
    ck[1, 3]=0.0
    ck[1, 4]=0.0
    ck[1, 5]=0.0
    ck[2, 0]=1.0
    ck[2, 1]=1.0
    ck[2, 2]=1.0
    ck[2, 3]=0.0
    ck[2, 4]=0.0
    ck[2, 5]=0.0
    ck[3, 0]=0.0
    ck[3, 1]=0.0
    ck[3, 2]=0.0
    ck[3, 3]=0.0
    ck[3, 4]=0.0
    ck[3, 5]=0.0
    ck[4, 0]=0.0
    ck[4, 1]=0.0
    ck[4, 2]=0.0
    ck[4, 3]=0.0
    ck[4, 4]=0.0
    ck[4, 5]=0.0
    ck[5, 0]=0.0
    ck[5, 1]=0.0
    ck[5, 2]=0.0
    ck[5, 3]=0.0
    ck[5, 5]=0.0
    ck[5, 5]=0.0

    cmu[0, 0]=4.0/3.0
    cmu[0, 1]=-2.0/3.0
    cmu[0, 2]=-2.0/3.0
    cmu[0, 3]=0.0
    cmu[0, 4]=0.0
    cmu[0, 5]=0.0
    cmu[1, 0]=-2.0/3.0
    cmu[1, 1]=4.0/3.0
    cmu[1, 2]=-2.0/3.0
    cmu[1, 3]=0.0
    cmu[1, 4]=0.0
    cmu[1, 5]=0.0
    cmu[2, 0]=-2.0/3.0
    cmu[2, 1]=-2.0/3.0
    cmu[2, 2]=4.0/3.0
    cmu[2, 3]=0.0
    cmu[2, 4]=0.0
    cmu[2, 5]=0.0
    cmu[3, 0]=0.0
    cmu[3, 1]=0.0
    cmu[3, 2]=0.0
    cmu[3, 3]=1.0
    cmu[3, 4]=0.0
    cmu[3, 5]=0.0
    cmu[4, 0]=0.0
    cmu[4, 1]=0.0
    cmu[4, 2]=0.0
    cmu[4, 3]=0.0
    cmu[4, 4]=1.0
    cmu[4, 5]=0.0
    cmu[5, 0]=0.0
    cmu[5, 1]=0.0
    cmu[5, 2]=0.0
    cmu[5, 3]=0.0
    cmu[5, 4]=0.0
    cmu[5, 5]=1.0
    
    cmod = np.zeros((nphase, 6, 6), dtype=np.float_)
    for k in range(nphase):
        for j in range(6):
            for i in range(6):
                cmod[k, i, j] = phasemod[k, 0]*ck[i, j] + phasemod[k, 1]*cmu[i,j]
    
    #  set up Simpson's integration rule weight vector
    g = np.zeros((3, 3, 3), dtype=np.float_)
    for k in range(3):
        for j in range(3):
            for i in range(3):
                nm = 0
                if i == 1:
                    nm = nm + 1
                if j == 1:
                    nm = nm + 1
                if k == 1:
                    nm = nm + 1
                g[i, j, k] = 4**nm
    
    #  loop over the nphase kinds of pixels and Simpson's rule quadrature
    #  points in order to compute the stiffness matrices.  Stiffness matrices
    #  of trilinear finite elements are quadratic in x, y, and z, so that
    #  Simpson's rule quadrature gives exact results.
    dndx = np.zeros(8, dtype=np.float_)
    dndy = np.zeros(8, dtype=np.float_)
    dndz = np.zeros(8, dtype=np.float_)
    
    for ijk in range(nphase):
        for k in range(1,4):
            for j in range(1,4):
                for i in range(1,4):
                    
                    x = (i-1)/2.0
                    y = (j-1)/2.0
                    z = (k-1)/2.0
                    
                    #  dndx means the negative derivative, with respect to x, of the shape
                    #  matrix N (see manual, Sec. 2.2), dndy, and dndz are similar.
                    
                    dndx[0]=-(1.0-y)*(1.0-z)
                    dndx[1]=(1.0-y)*(1.0-z)
                    dndx[2]=y*(1.0-z)
                    dndx[3]=-y*(1.0-z)
                    dndx[4]=-(1.0-y)*z
                    dndx[5]=(1.0-y)*z
                    dndx[6]=y*z
                    dndx[7]=-y*z
                    dndy[0]=-(1.0-x)*(1.0-z)
                    dndy[1]=-x*(1.0-z)
                    dndy[2]=x*(1.0-z)
                    dndy[3]=(1.0-x)*(1.0-z)
                    dndy[4]=-(1.0-x)*z
                    dndy[5]=-x*z
                    dndy[6]=x*z
                    dndy[7]=(1.0-x)*z
                    dndz[0]=-(1.0-x)*(1.0-y)
                    dndz[1]=-x*(1.0-y)
                    dndz[2]=-x*y
                    dndz[3]=-(1.0-x)*y
                    dndz[4]=(1.0-x)*(1.0-y)
                    dndz[5]=x*(1.0-y)
                    dndz[6]=x*y
                    dndz[7]=(1.0-x)*y
                    
                    #  now build strain matrix
                    es = np.zeros((6, 8, 3))
                    
                    for n in range(8):
                        es[0,n,0]=dndx[n]
                        es[1,n,1]=dndy[n]
                        es[2,n,2]=dndz[n]
                        es[3,n,0]=dndz[n]
                        es[3,n,2]=dndx[n]
                        es[4,n,1]=dndz[n]
                        es[4,n,2]=dndy[n]
                        es[5,n,0]=dndy[n]
                        es[5,n,1]=dndx[n]
                        
                    #  Matrix multiply to determine value at (x,y,z), multiply by
                    #  proper weight, and sum into dk, the stiffness matrix
                    for mm in range(3):
                        for nn in range(3):
                            for ii in range(8):
                                for jj in range(8):     
                                    #  Define sum over strain matrices and elastic moduli matrix for
                                    #  stiffness matrix
                                    # Fortran uses "sum"
                                    sum0 = 0.0
                                    for kk in range(6):
                                        for ll in range(6):
                                            sum0 = sum0 + es[kk,ii,mm]*cmod[ijk,kk,ll]*es[ll,jj,nn]
                                    dk[ijk,ii,mm,jj,nn] = dk[ijk,ii,mm,jj,nn] + g[i-1,j-1,k-1]*sum0/216.0
                    
    #  Set up vector for linear term, b, and constant term, C,
    #  in the elastic energy.  This is done using the stiffness matrices,
    #  and the periodic terms in the applied strain that come in at the 
    #  boundary pixels via the periodic boundary conditions and the 
    #  condition that an applied macroscopic strain exists (see Sec. 2.2 
    #  in the manual). It is easier to set b up this way than to analytically
    #  write out all the terms involved.
    
    #  Initialize b and C  
    
    b = np.zeros((ns, 3), dtype=np.float_)
    C = 0.0

    #  For all cases, the correspondence between 1-8 finite element node
    #  labels and 1-27 neighbor labels is (see Table 4 in manual):  
    #  1:ib(m,27), 2:ib(m,3),
    #  3:ib(m,2),4:ib(m,1),
    #  5:ib(m,26),6:ib(m,19)
    #  7:ib(m,18),8:ib(m,17) 
    
    #Fortran uses "is"
    is0 = np.zeros(8, dtype=np.int32)
    
    is0[0]=26
    is0[1]=2
    is0[2]=1
    is0[3]=0
    is0[4]=25
    is0[5]=18
    is0[6]=17
    is0[7]=16

    #  x=nx face
    delta = np.zeros((8, 3), dtype=np.float_)
    
    for i3 in range(3):
        for i8 in range(8):
            delta[i8, i3] = 0.0
            if i8 == 1 or i8 == 2 or i8 == 5 or i8 == 6:
                delta[i8, 0] = exx*nx
                delta[i8, 1] = exy*nx
                delta[i8, 2] = exz*nx

    for j in range(1, ny):
        for k in range(1, nz):
            m = nxy*(k-1)+j*nx-1
            for nn in range(3):
                for mm in range(8):
                    #Fortran uses "sum"
                    sum0 = 0.0
                    for m3 in range(3):
                        for m8 in range(8):
                            sum0 = sum0 + delta[m8,m3]*dk[pix[m],m8,m3,mm,nn]
                            C = C + 0.5*delta[m8,m3]*dk[pix[m],m8,m3,mm,nn]*delta[mm,nn]
                    b[ib[m,is0[mm]],nn] = b[ib[m,is0[mm]],nn] + sum0

    #  y=ny face
    delta = np.zeros((8, 3), dtype=np.float_)
    
    for i3 in range(3):
        for i8 in range(8):
            delta[i8, i3] = 0.0
            if i8 == 2 or i8 == 3 or i8 == 6 or i8 == 7:
                delta[i8, 0] = exy*ny
                delta[i8, 1] = eyy*ny
                delta[i8, 2] = eyz*ny
    
    for i in range(1, nx):
        for k in range(1, nz):
            m = nxy*(k-1) + nx*(ny-1) + i - 1
            for nn in range(3):
                for mm in range(8):
                    #Fortran uses "sum"
                    sum0 = 0.0
                    for m3 in range(3):
                        for m8 in range(8):
                            sum0 = sum0 + delta[m8,m3]*dk[pix[m],m8,m3,mm,nn]
                            C = C + 0.5*delta[m8,m3]*dk[pix[m],m8,m3,mm,nn]*delta[mm,nn]
                    b[ib[m,is0[mm]],nn] = b[ib[m,is0[mm]],nn] + sum0

    #  z=nz face
    delta = np.zeros((8, 3), dtype=np.float_)
    
    for i3 in range(3):
        for i8 in range(8):
            delta[i8, i3] = 0.0
            if i8 == 4 or i8 == 5 or i8 == 6 or i8 == 7:
                delta[i8, 0] = exz*nz
                delta[i8, 1] = eyz*nz
                delta[i8, 2] = ezz*nz
                
    for i in range(1, nx):
        for j in range(1, ny):
            m = nxy*(nz-1)+nx*(j-1)+i-1
            for nn in range(3):
                for mm in range(8):
                    #Fortran uses "sum"
                    sum0 = 0.0
                    for m3 in range(3):
                        for m8 in range(8):
                            sum0 = sum0 + delta[m8,m3]*dk[pix[m],m8,m3,mm,nn]
                            C = C + 0.5*delta[m8,m3]*dk[pix[m],m8,m3,mm,nn]*delta[mm,nn]
                    b[ib[m,is0[mm]],nn] = b[ib[m,is0[mm]],nn] + sum0

    #  x=nx y=ny edge
    delta = np.zeros((8, 3), dtype=np.float_)
    
    for i3 in range(3):
        for i8 in range(8):
            delta[i8,i3] = 0.0
            if (i8 == 1 or i8 == 5):
                delta[i8,0]=exx*nx
                delta[i8,1]=exy*nx
                delta[i8,2]=exz*nx
            if (i8 == 3 or i8 == 7):
                delta[i8,0]=exy*ny
                delta[i8,1]=eyy*ny
                delta[i8,2]=eyz*ny
            if (i8 == 2 or i8 == 6):
                delta[i8,0]=exy*ny+exx*nx
                delta[i8,1]=eyy*ny+exy*nx
                delta[i8,2]=eyz*ny+exz*nx

    for k in range(1, nz):
        m = nxy*k - 1 
        for nn in range(3):
            for mm in range(8):
                sum0 = 0.0
                for m3 in range(3):
                    for m8 in range(8):
                        sum0 = sum0 + delta[m8,m3]*dk[pix[m],m8,m3,mm,nn]
                        C = C + 0.5*delta[m8,m3]*dk[pix[m],m8,m3,mm,nn]*delta[mm,nn]       
                b[ib[m,is0[mm]],nn] = b[ib[m,is0[mm]],nn] + sum0

    # x=nx z=nz edge
    delta = np.zeros((8, 3), dtype=np.float_)
    
    for i3 in range(3):
        for i8 in range(8):
            delta[i8,i3] = 0.0
            if (i8 == 1 or i8 == 2):
                delta[i8,0]=exx*nx
                delta[i8,1]=exy*nx
                delta[i8,2]=exz*nx
            if (i8 == 4 or i8 == 7):
                delta[i8,0]=exz*nz
                delta[i8,1]=eyz*nz
                delta[i8,2]=ezz*nz
            if (i8 == 5 or i8 == 6):
                delta[i8,0]=exz*nz+exx*nx
                delta[i8,1]=eyz*nz+exy*nx
                delta[i8,2]=ezz*nz+exz*nx

    for j in range(1, ny):
        m = nxy*(nz-1)+nx*(j-1)+nx - 1
        for nn in range(3):
            for mm in range(8):
                sum0 = 0.0
                for m3 in range(3):
                    for m8 in range(8):
                        sum0 = sum0 + delta[m8,m3]*dk[pix[m],m8,m3,mm,nn]
                        C = C + 0.5*delta[m8,m3]*dk[pix[m],m8,m3,mm,nn]*delta[mm,nn]       
                b[ib[m,is0[mm]],nn] = b[ib[m,is0[mm]],nn] + sum0

    # y=ny z=nz edge
    delta = np.zeros((8, 3), dtype=np.float_)
    
    for i3 in range(3):
        for i8 in range(8):
            delta[i8,i3] = 0.0
            if (i8 == 4 or i8 == 5):
                delta[i8,0]=exz*nz
                delta[i8,1]=eyz*nz
                delta[i8,2]=ezz*nz
            if (i8 == 2 or i8 == 3):
                delta[i8,0]=exy*ny
                delta[i8,1]=eyy*ny
                delta[i8,2]=eyz*ny
            if (i8 == 6 or i8 == 7):
                delta[i8,0]=exy*ny+exz*nz
                delta[i8,1]=eyy*ny+eyz*nz
                delta[i8,2]=eyz*ny+ezz*nz

    for i in range(1, nx):
        m=nxy*(nz-1)+nx*(ny-1)+i - 1
        for nn in range(3):
            for mm in range(8):
                sum0 = 0.0
                for m3 in range(3):
                    for m8 in range(8):
                        sum0 = sum0 + delta[m8,m3]*dk[pix[m],m8,m3,mm,nn]
                        C = C + 0.5*delta[m8,m3]*dk[pix[m],m8,m3,mm,nn]*delta[mm,nn]       
                b[ib[m,is0[mm]],nn] = b[ib[m,is0[mm]],nn] + sum0
                
    # x=nx y=ny z=nz corner
    delta = np.zeros((8, 3), dtype=np.float_)
    
    for i3 in range(3):
        for i8 in range(8):
            delta[i8,i3] = 0.0
            if(i8==1):
                delta[i8,0]=exx*nx
                delta[i8,1]=exy*nx
                delta[i8,2]=exz*nx
            
            if(i8==3):
                delta[i8,0]=exy*ny
                delta[i8,1]=eyy*ny
                delta[i8,2]=eyz*ny
             
            if(i8==4):
                delta[i8,0]=exz*nz
                delta[i8,1]=eyz*nz
                delta[i8,2]=ezz*nz
             
            if(i8==7):
                delta[i8,0]=exy*ny+exz*nz
                delta[i8,1]=eyy*ny+eyz*nz
                delta[i8,2]=eyz*ny+ezz*nz
             
            if(i8==5):
                delta[i8,0]=exx*nx+exz*nz
                delta[i8,1]=exy*nx+eyz*nz
                delta[i8,2]=exz*nx+ezz*nz
             
            if(i8==2):
                delta[i8,0]=exx*nx+exy*ny
                delta[i8,1]=exy*nx+eyy*ny
                delta[i8,2]=exz*nx+eyz*ny
            
            if(i8==6):
                delta[i8,0] = exx*nx+exy*ny+exz*nz
                delta[i8,1] = exy*nx+eyy*ny+eyz*nz
                delta[i8,2] = exz*nx+eyz*ny+ezz*nz
    m = nx*ny*nz - 1
    for nn in range(3):
        for mm in range(8):
            sum0 = 0.0
            for m3 in range(3):
                for m8 in range(8):
                    sum0 = sum0 + delta[m8,m3]*dk[pix[m],m8,m3,mm,nn]
                    C = C + 0.5*delta[m8,m3]*dk[pix[m],m8,m3,mm,nn]*delta[mm,nn]       
            b[ib[m,is0[mm]],nn] = b[ib[m,is0[mm]],nn] + sum0    

    return C, dk, b, cmod

if __name__ == "__main__":
    nx = 10
    ny = 20
    nz = 30
    ns = nx * ny * nz
    exx = 1.0
    eyy = 1.0
    ezz = 1.0
    exz = 1.0/2
    eyz = 2.0/2
    exy = 3.0/2
    
    nphase = 2
    phasemod = np.array([[1, 1], [2, 2]])
    pix = np.random.randint(low = 0, high = 2, size = (10, 20, 30)).flatten(order = 'F')
    ib = 0
    femat(nx,ny,nz,ns,exx,eyy,ezz,exz,eyz,exy,phasemod,nphase,pix,ib)
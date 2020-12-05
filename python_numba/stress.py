# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 01:39:06 2020

@author: Ken Ikeda
@email: ikeda.ken@utexas.edu

Ph.D. candidate 
Jackson School of Geosciences
The University of Texas at Austin, Texas, USA
"""
import numpy as np
from numba import jit

@jit(nopython = True)
def stress(nx, ny, nz, ns, exx, eyy, exz, eyz, ezz, exy, u, ib, pix, cmod):
    nxy = nx*ny
     
    #  set up single element strain matrix
    #  dndx, dndy, and dndz are the components of the average strain 
    #  matrix in a pixel

    dndx = np.zeros(8, dtype=np.float_)
    dndy = np.zeros(8, dtype=np.float_)
    dndz = np.zeros(8, dtype=np.float_)
    
    dndx[0]=-0.25
    dndx[1]=0.25
    dndx[2]=0.25
    dndx[3]=-0.25
    dndx[4]=-0.25
    dndx[5]=0.25
    dndx[6]=0.25
    dndx[7]=-0.25
    dndy[0]=-0.25
    dndy[1]=-0.25
    dndy[2]=0.25
    dndy[3]=0.25
    dndy[4]=-0.25
    dndy[5]=-0.25
    dndy[6]=0.25
    dndy[7]=0.25
    dndz[0]=-0.25
    dndz[1]=-0.25
    dndz[2]=-0.25
    dndz[3]=-0.25
    dndz[4]=0.25
    dndz[5]=0.25
    dndz[6]=0.25
    dndz[7]=0.25
    
    #  Build averaged strain matrix, follows code in femat, but for average
    #  strain over the pixel, not the strain at a point.
    
    es = np.zeros((6, 8, 3), dtype=np.float_)
    
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
    
    #  Compute components of the average stress and strain tensors in each pixel
    strxx = 0.0
    stryy = 0.0
    strzz = 0.0
    strxz = 0.0
    stryz = 0.0
    strxy = 0.0
    
    sxx = 0.0
    syy = 0.0
    szz = 0.0
    sxz = 0.0
    syz = 0.0
    sxy = 0.0   
    
    uu = np.zeros((8, 3), dtype=np.float_)
    m = np.arange(0, ns)
    
    for k in range(1, nz+1):
        for j in range(1, ny+1):
            for i in range(1, nx+1):
                m = (k-1)*nxy + (j-1)*nx + i - 1
                #  load in elements of 8-vector using pd. bd. conds.
                for mm in range(3):
                    uu[0,mm] = u[m,mm] 
                    uu[1,mm] = u[ib[m,2],mm]
                    uu[2,mm] = u[ib[m,1],mm]
                    uu[3,mm] = u[ib[m,0],mm]
                    uu[4,mm] = u[ib[m,25],mm]
                    uu[5,mm] = u[ib[m,18],mm]
                    uu[6,mm] = u[ib[m,17],mm]
                    uu[7,mm] = u[ib[m,16],mm]
                    
                #  Correct for periodic boundary conditions, some displacements are wrong
                #  for a pixel on a periodic boundary.  Since they come from an opposite 
                #  face, need to put in applied strain to correct them.
                
                if i==nx:
                    uu[1,0]=uu[1,0]+exx*nx
                    uu[1,1]=uu[1,1]+exy*nx
                    uu[1,2]=uu[1,2]+exz*nx
                    uu[2,0]=uu[2,0]+exx*nx
                    uu[2,1]=uu[2,1]+exy*nx
                    uu[2,2]=uu[2,2]+exz*nx
                    uu[5,0]=uu[5,0]+exx*nx
                    uu[5,1]=uu[5,1]+exy*nx
                    uu[5,2]=uu[5,2]+exz*nx
                    uu[6,0]=uu[6,0]+exx*nx
                    uu[6,1]=uu[6,1]+exy*nx
                    uu[6,2]=uu[6,2]+exz*nx
                    
                if j==ny:
                    uu[2,0]=uu[2,0]+exy*ny
                    uu[2,1]=uu[2,1]+eyy*ny
                    uu[2,2]=uu[2,2]+eyz*ny
                    uu[3,0]=uu[3,0]+exy*ny
                    uu[3,1]=uu[3,1]+eyy*ny
                    uu[3,2]=uu[3,2]+eyz*ny
                    uu[6,0]=uu[6,0]+exy*ny
                    uu[6,1]=uu[6,1]+eyy*ny
                    uu[6,2]=uu[6,2]+eyz*ny
                    uu[7,0]=uu[7,0]+exy*ny
                    uu[7,1]=uu[7,1]+eyy*ny
                    uu[7,2]=uu[7,2]+eyz*ny
                
                if k==nz:
                    uu[4,0]=uu[4,0]+exz*nz
                    uu[4,1]=uu[4,1]+eyz*nz
                    uu[4,2]=uu[4,2]+ezz*nz
                    uu[5,0]=uu[5,0]+exz*nz
                    uu[5,1]=uu[5,1]+eyz*nz
                    uu[5,2]=uu[5,2]+ezz*nz
                    uu[6,0]=uu[6,0]+exz*nz
                    uu[6,1]=uu[6,1]+eyz*nz
                    uu[6,2]=uu[6,2]+ezz*nz
                    uu[7,0]=uu[7,0]+exz*nz
                    uu[7,1]=uu[7,1]+eyz*nz
                    uu[7,2]=uu[7,2]+ezz*nz
            
                #  local stresses and strains in a pixel        
                s11 = np.sum(es[0, :, :] * uu)
                s22 = np.sum(es[1, :, :] * uu)
                s33 = np.sum(es[2, :, :] * uu)
                s13 = np.sum(es[3, :, :] * uu)
                s23 = np.sum(es[4, :, :] * uu)
                s12 = np.sum(es[5, :, :] * uu)
                
                str11=0.0
                str22=0.0
                str33=0.0
                str13=0.0
                str23=0.0
                str12=0.0
                
                for n in range(6):
                    str11=str11+np.sum(cmod[pix[m],0,n]*es[n,:,:]*uu)
                    str22=str22+np.sum(cmod[pix[m],1,n]*es[n,:,:]*uu)
                    str33=str33+np.sum(cmod[pix[m],2,n]*es[n,:,:]*uu)
                    str13=str13+np.sum(cmod[pix[m],3,n]*es[n,:,:]*uu)
                    str23=str23+np.sum(cmod[pix[m],4,n]*es[n,:,:]*uu)
                    str12=str12+np.sum(cmod[pix[m],5,n]*es[n,:,:]*uu)
                    
                #  sum local strains and stresses into global values
                strxx=strxx+str11
                stryy=stryy+str22
                strzz=strzz+str33
                strxz=strxz+str13
                stryz=stryz+str23
                strxy=strxy+str12
                sxx=sxx+s11
                syy=syy+s22
                szz=szz+s33
                sxz=sxz+s13
                syz=syz+s23
                sxy=sxy+s12
                
    #  Volume average of global stresses and strains
    strxx=strxx/ns
    stryy=stryy/ns
    strzz=strzz/ns
    strxz=strxz/ns
    stryz=stryz/ns
    strxy=strxy/ns
    sxx=sxx/ns
    syy=syy/ns
    szz=szz/ns
    sxz=sxz/ns
    syz=syz/ns
    sxy=sxy/ns
    return strxx, stryy, strzz, strxz, stryz, strxy, sxx, syy, szz, sxz, syz, sxy
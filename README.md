# elas3D-python
A translated code of elas3D from NIST

elas3D is a Fortran77 code written by Garboczi (1978) from National Institute of Standards and Technology (NIST). The code solves the linear elasticity equation with Finite Element approximation. Given a discrete cube of size Nx-by-Ny-by-Nz and a prescribed strain boundary condition (exx eyy exx eyz exz exy), the code solves for stresses at each node in the domain that minimized the energy of the system. Note that the algorithm uses the periodic boundary condition. 

Citation, code, and manual should be refered to https://www.nist.gov/services-resources/software/finite-elementfinite-difference-programs 

Here, I rewrite the Fortran code using python 3 for accessibility and convenience. This reporsitory consists of three parts:

* -- master
  * |-- python original
  * |-- python numpy
  * |-- python numba

The "python original" contains 1-to-1 translation from the Fortran code with the help of the numpy package to define multidimensional arrays. The "python numpy" uses numpy optimization techniques such as index slicing to speed up the computational time. The "python numba" is an upgraded version of the previous one with numba Just-In-Time (JIT) compiler. In term of speed, "python numba" would be the fastest one (excluding compilation time). 

As we would like to maintain the originality of the code, the inputs of the code are:

1) microstructure.dat --> a file contains a list of interger (starting from 0) that describes the structures of the material. Each number represent different phase. 

2) nx, ny, nz --> the size of the material (in pixels).

3) nphase --> the total number of phases in the system. Note that the maximum number in "microstructure.dat" must not exceed (nphase-1). 

4) phasemod --> a 2D array where each row represent elastic material of a phase. Note that we assume each phase to be isotropic. Therefore, we only need two elastic constants: Young's modulus (first column) and Poisson ratio (second column). 

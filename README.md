# elas3D-python-matlab
A translated code of "elas3D" from NIST. 

elas3D is a Fortran77 code written by Garboczi (1978) from National Institute of Standards and Technology (NIST). The code solves the linear elasticity equation with Finite Element approximation. Given a discrete cube of size Nx-by-Ny-by-Nz and a prescribed strain boundary condition (exx eyy exx eyz exz exy), the code solves for stresses at each node in the domain that minimized the energy of the system. Note that the algorithm uses the periodic boundary condition. 

Citation, code, and manual of the original code should be referred to https://www.nist.gov/services-resources/software/finite-elementfinite-difference-programs. The MATLAB code, presented in this repository, is used in

The modified version of this code has been used in the following publications:
1) Ken Ikeda, Eric Goldfarb, and Nicola Tisato, (2017), "*Static elastic properties of Berea sandstone by means of segmentation-less digital rock physics*," SEG Technical Program Expanded Abstracts : 3914-3919. https://doi.org/10.1190/segam2017-17789805.1
2)  Ken Ikeda, Shankar Subramaniyan, Beatriz Quintal, Eric J. Goldfarb, Erik H. Saenger, and Nicola Tisato,  (2021), "*Low-frequency elastic properties of a polymineralic carbonate: laboratory measurement and digital rock physics*," Frontier in Earth Sciences, doi: 10.3389/feart.2021.628544 

## Modification to the original code 
I rewrite the code for accessibility and convenience. Both python 3 and MATLAB codes will be organized in the following manner, 

* -- master
    *   python
        *  python_original
        *  python_numpy
        *  python_numba
    *  matlab
        *  matlab_original 
        *  matlab_softVectorized

*python_original* and *matlab_original* contain almost 1-to-1 translation of the original elas3D code. In python, I use the Numpy package to define multidimensional arrays. The modifications on those two files are in defining variables (initializing variables). For example, I change 
```Fortran
do 2090 m3=1,3
do 2090 m=1,ns
gb(m,m3)=0.0
2090	continue
```
to
```python
gb = numpy.zeros((ns, 3))
```

*python_numpy* and *matlab_softVectorized* use vectorization techniques (implicit parallelization in MATLAB) to speed up the code. For example, I change 

```Fortran
utot=C
do 3100 m3=1,3
do 3100 m=1,ns
utot=utot+0.5*u(m,m3)*gb(m,m3)+b(m,m3)*u(m,m3)
gb(m,m3)=gb(m,m3)+b(m,m3)
3100	continue
```
to
```python
utot = C + 0.5*numpy.sum(u*gb) + np.sum(b*u)
gb = gb + b
```

Lastly, *python_numba* is the same as *python_numpy* but includes the Just-In-Time (JIT) compiler from the Numba package. As of today, MATLAB already uses JIT compiler.  

Users can start modifying the provided codes to fit their needs. Optimization could be done in many different ways to speed up the code. For example,

1.	Explicitly parallelize the code
2.	Convert the code into C with MATLAB MEX file
3.	Cut down unnecessary computations. To be specific, most of the time, we are dealing with isotropic materials. Therefore, the calculation in the codes will contain many “zeros”. Try to run the *stress* function.
4.	Instead of reading the data from *microstructure.dat*, you can read the data directly on to the program. 

If you want to display the final stress field map, modify the *stress* function such that, at each iteration, the program output str11, str22 ... .
% *********************** elas3d.f ***********************************
%% Convert from Fortran 77 to MATLAB : Ken Ikeda
% Date: Nov. 2, 2016
% The author (me) hasn't change any computational algorithms in the
% following code. All copyrights belong to their respective owners (NIST).   
%% ***** BACKGROUND *****
% This program solves the linear elastic equations in a
% random linear elastic material, subject to an applied macroscopic strain,
% using the finite element method. Each pixel in the 3-D digital
% image is a cubic tri-linear finite element, having its own
% elastic moduli tensor. Periodic boundary conditions are maintained.
% In the comments below, (USER) means that this is a section of code that
% the user might have to change for his particular problem. Therefore the
% user is encouraged to search for this string.
% ***** PROBLEM AND VARIABLE DEFINITION *****
% The problem being solved is the minimization of the energy
% 1/2 uAu + b u + C, where A is the Hessian matrix composed of the
% stiffness matrices (dk) for each pixel/element, b is a constant vector
% and C is a constant that are determined by the applied strain and
% the periodic boundary conditions, and u is a vector of
% all the displacements. The solution
% method used is the conjugate gradient relaxation algorithm.
% Other variables are: gb is the gradient = Au+b, h and Ah are
% auxiliary variables used in the conjugate gradient algorithm (in dembx),
% dk(n,i,j) is the stiffness matrix of the n'th phase, cmod(n,i,j) is
% the elastic moduli tensor of the n'th phase, pix is a vector that gives
% the phase label of each pixel, ib is a matrix that gives the labels of
% the 27 (counting itself) neighbors of a given node, prob is the volume
% fractions of the various phases,
% strxx, stryy, strzz, strxz, stryz, and strxy are the six Voigt
% volume averaged total stresses, and
% sxx, syy, szz, sxz, syz, and sxy are the six Voigt
% volume averaged total strains.
% ***** DIMENSIONS *****
% The vectors u,gb,b,h, and Ah are dimensioned to be the system size,
% ns=nx*ny*nz, with three components, where the digital image of the
% microstructure considered is a rectangular paralleliped, nx x ny x nz
% in size. The arrays pix and ib are are also dimensioned to the system size.
% The array ib has 27 components, for the 27 neighbors of a node.
% Note that the program is set up at present to have at most 100
% different phases. This can easily be changed, simply by changing
% the dimensions of dk, prob, and cmod. The parameter nphase gives the
% number of phases being considered in the problem.
% All arrays are passed between subroutines using simple common statements.
% STRONGLY SUGGESTED: READ THE MANUAL BEFORE USING PROGRAM!!
% (USER) Change these dimensions and in other subroutines at same time.
% For example, search and replace all occurrences throughout the
% program of "(8000" by "(64000", to go from a
% 20 x 20 x 20 system to a 40 x 40 x 40 system.

clear 
clc 

%% [USER] Define dimension nx,ny,nz 
nx=20;      
ny=10;      
nz=30;  

ns=nx*ny*nz;                 
fprintf('nx = %d ny = %d nz = %d ns = %d\n', nx, ny, nz, ns)
%% [USER] nphase is the number of phases being considered in the problem.
% The values of pix(m) will run from 1 to nphase.
nphase = 2;

%% [USER] gtest is the stopping criterion, the number to which the quantity gg=gb*gb is compared.
% Usually gtest = abc*ns, so that when gg < gtest, the rms value per pixel of gb is less than sqrt(abc).
gtest = 1.e-8*ns;

%% [USER] The parameter phasemod(i,j) is the bulk (i,1) and shear (i,2) moduli of
% the i'th phase. These can be input in terms of Young's moduli E(i,1) and
% Poisson's ratio nu (i,2). The program, in do loop 1144, then changes them
% to bulk and shear moduli, using relations for isotropic elastic
% moduli. For anisotropic elastic material, one can directly input
% the elastic moduli tensor cmod in subroutine femat, and skip this part.
% If you wish to input in terms of bulk (1) and shear (2), then make sure
% to comment out the do 1144 loop.
phasemod(1,1) = 1.0;          
phasemod(1,2) = 0.2;
phasemod(2,1) = 0.5;          
phasemod(2,2) = 0.2;
%% [USER]  Program uses bulk modulus (1) and shear modulus (2), so transform
% Young's modulis (1) and Poisson's ratio (2).
for i=1:nphase
    E = phasemod(i,1);
    phasemod(i,1) = phasemod(i,1)/ (3 * (1. - 2.*phasemod(i, 2)));
    phasemod(i,2) = E / (2 * (1 + phasemod(i,2)));
end
%% Construct the neighbor table, ib(m,n)
% First construct the 27 neighbor table in terms of delta i, delta j, and
% delta k information (see Table 3 in manual)

in(1)=0;
in(2)=1;
in(3)=1;
in(4)=1;
in(5)=0;
in(6)=-1;
in(7)=-1;
in(8)=-1;

jn(1)=1;
jn(2)=1;
jn(3)=0;
jn(4)=-1;
jn(5)=-1;
jn(6)=-1;
jn(7)=0;
jn(8)=1;

for n=1:8
    kn(n)=0;
    kn(n+8)=-1;
    kn(n+16)=1;
    in(n+8)=in(n);
    in(n+16)=in(n);
    jn(n+8)=jn(n);
    jn(n+16)=jn(n);
end

in(25)=0;
in(26)=0;
in(27)=0;
jn(25)=0;
jn(26)=0;
jn(27)=0;
kn(25)=-1;
kn(26)=1;
kn(27)=0;

% Now construct neighbor table according to 1-d labels
% Matrix ib(m,n) gives the 1-d label of the n'th neighbor (n=1,27) of
% the node labelled m.

ib = zeros(ns, 27);
nxy = nx*ny;
for k=1:nz
    for j=1:ny
        for i=1:nx
            m = nxy*(k-1)+nx*(j-1)+i;
            for n=1:27
                i1=i+in(n);
                j1=j+jn(n);
                k1=k+kn(n);
                        if i1 < 1     
                            i1 = i1 + nx; end
                        if i1 > nx   
                            i1 = i1 - nx; end
                        if j1 < 1    
                            j1 = j1 + ny; end
                        if j1 > ny    
                            j1 = j1 - ny; end
                        if k1 < 1    
                            k1 = k1 + nz; end
                        if k1 > nz   
                            k1 = k1 - nz; end
                m1 = nxy*(k1-1)+nx*(j1-1)+i1;
                ib(m,n) = m1;
            end
        end
    end
end    
%% Compute the average stress and strain in each microstructure.
% [USER] npoints is the number of microstructures to use.
npoints = 1;

for micro=1:npoints
    % Read in a microstructure in subroutine ppixel, and set up pix(m)
    % with the appropriate phase assignments.
    pix = ppixel(nx, ny, nz, ns, nphase);

    % Count and output the volume fractions of the different phases
    [prob] = assig(ns, nphase, pix);
    
    for i=1:nphase
        fprintf('Phase %d bulk = %12.6f shear = %12.5f \n',i,phasemod(i,1),phasemod(i,2))
    end
    
    for i=1:nphase
        fprintf('Volume fraction of phase %d is %8.5f \n',i, prob(i))            
    end
    
    % [USER] Set applied strains
    % Actual shear strain applied in do 1050 loop is exy, exz, and eyz as
    % given in the statements below. The engineering shear strain, by which
    % the shear modulus is usually defined, is twice these values.
    exx=0.1;        eyy=0.1;        ezz=0.1;
    exz=0.1/2;      eyz=0.2/2;      exy=0.3/2;
    fprintf('Applied engineering strains \n exx eyy ezz exz eyz exy \n %.2f %.2f %.2f %.2f %.2f %.2f \n',exx,eyy,ezz,2.*exz,2.*eyz,2.*exy)

    % Set up the elastic modulus variables, finite element stiffness matrices,
    % the constant, C, and vector, b, required for computing the energy.
    % [USER] If anisotropic elastic moduli tensors are used, these need to be
    % input in subroutine femat.
    [C,dk,b,cmod] = femat(nx,ny,nz,ns,exx,eyy,ezz,exz,eyz,exy,phasemod,nphase,pix,ib);
        
    % Apply chosen strains as a homogeneous macroscopic strain as the initial condition.
    u = zeros(ns, 3);
    
    for k=1:nz
        for j=1:ny
            for i=1:nx
                m=nxy*(k-1)+nx*(j-1)+i;
                x=i-1;
                y=j-1;
                z=k-1;
                u(m,1)=x*exx+y*exy+z*exz;
                u(m,2)=x*exy+y*eyy+z*eyz;
                u(m,3)=x*exz+y*eyz+z*ezz;
            end
        end
    end           
    
    % RELAXATION LOOP
    % [USER] kmax is the maximum number of times dembx will be called, with
    % ldemb conjugate gradient steps performed during each call. The total
    % number of conjugate gradient steps allowed for a given elastic
    % computation is kmax*ldemb.
    kmax = 40;
    ldemb = 50; 
    ltot = 0;
    Lstep = 0;
    h = 0;  %initialization
    
    % Call energy to get initial energy and initial gradient
    [gb,utot] = energy(ns,C,ib,u,pix,dk,b);

    % gg is the norm squared of the gradient (gg=gb*gb)
    gg = 0;
    
    for m3 = 1:3
        for m = 1:ns
            gg=gg+gb(m,m3)*gb(m,m3);
        end
    end
    
    fprintf('Initial energy = %.4f gg = %.4f\n',utot,gg)

    for kkk=1:kmax
        [Ah, Lstep, h, u, gg] = dembx(ns, Lstep, gg, dk, gtest, ldemb, kkk, gb, ib, pix, u, h);
        ltot = ltot + Lstep;
        
        % Call energy to compute energy after dembx call. If gg < gtest, this
        % will be the final energy. If gg is still larger than gtest, then this
        % will give an intermediate energy with which to check how the
        % relaxation process is coming along.
         
        [gb,utot] = energy(ns,C,ib,u,pix,dk,b);
  
        fprintf('Energy = %.6f \t gg = %.6f \n Number of conjugate steps = %d\n',utot,gg,ltot)                
                          
        % If relaxation process is finished, jump out of loop
        if(gg<gtest) 
            break
        end
        % If relaxation process will continue, compute and output stresses
        % and strains as an additional aid to judge how the
        % relaxation procedure is progressing.
      
        [strxx,stryy,strzz,strxz,stryz,strxy,sxx,syy,szz,sxz,syz,sxy] = stress(nx,ny,nz,ns,exx,eyy,ezz,exz,eyz,exy,u,ib,pix,cmod);
        
        fprintf('stresses: xx,yy,zz,xz,yz,xy = %.5f %.5f %.5f %.5f %.5f %.5f\n',strxx,stryy,strzz,strxz,stryz,strxy)                
        fprintf('strains: xx,yy,zz,xz,yz,xy = %.5f %.5f %.5f %.5f %.5f %.5f\n',sxx,syy,szz,sxz,syz,sxy)                
        
    end
    
    [strxx,stryy,strzz,strxz,stryz,strxy,sxx,syy,szz,sxz,syz,sxy] = stress(nx,ny,nz,ns,exx,eyy,ezz,exz,eyz,exy,u,ib,pix,cmod);

    fprintf('stresses: xx,yy,zz,xz,yz,xy = %.5f %.5f %.5f %.5f %.5f %.5f\n',strxx,stryy,strzz,strxz,stryz,strxy)                
    fprintf('strains: xx,yy,zz,xz,yz,xy = %.5f %.5f %.5f %.5f %.5f %.5f\n',sxx,syy,szz,sxz,syz,sxy)                
        
end    

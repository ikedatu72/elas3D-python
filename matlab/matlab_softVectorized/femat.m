function [C,dk,b,cmod] = femat(nx,ny,nz,ns,exx,eyy,ezz,exz,eyz,exy,phasemod,nphase,pix,ib)
    % Subroutine that sets up the elastic moduli variables,
    % the stiffness matrices,dk, the linear term in
    % displacements, b, and the constant term, C, that appear in the total energy
    % due to the periodic boundary conditions

    nxy = nx*ny;
    % [USER] NOTE: complete elastic modulus matrix is used, so an anisotropic
    % matrix could be directly input at any point, since program is written
    % to use a general elastic moduli tensor, but is only explicitly
    % implemented for isotropic materials.

    % initialize stiffness matrices
    dk = zeros(nphase, 8, 3, 8, 3);

    % set up elastic moduli matrices for each kind of element
    % ck and cmu are the bulk and shear modulus matrices, which need to be
    % weighted by the actual bulk and shear moduli
    ck = zeros(6, 6);
    cmu = zeros(6, 6);

    ck = [1, 1, 1, 0, 0, 0;
          1, 1, 1, 0, 0, 0;
          1, 1, 1, 0, 0, 0;
          0, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0];

    cmu = [4/3, -2/3, -2/3, 0, 0, 0;
           -2/3, 4/3, -2/3, 0, 0, 0;
           -2/3, -2/3, 4/3, 0, 0, 0;
           0, 0, 0, 1, 0, 0;
           0, 0, 0, 0, 1, 0;
           0, 0, 0, 0, 0, 1];

    cmod = zeros(nphase, 6, 6);

    for k=1:nphase
        for j=1:6
            for i=1:6
                cmod(k,i,j)=phasemod(k,1)*ck(i,j)+phasemod(k,2)*cmu(i,j);
            end
        end
    end

    % set up Simpson's integration rule weight vector
    g = zeros(3, 3, 3);

    for k=1:3
        for j=1:3
            for i=1:3
                nm=0;
                if i==2     nm=nm+1;end
                if j==2     nm=nm+1;end
                if k==2     nm=nm+1;end
                g(i,j,k)=4.0^nm;
            end
        end    
    end        

    % loop over the nphase kinds of pixels and Simpson's rule quadrature
    % points in order to compute the stiffness matrices. Stiffness matrices
    % of trilinear finite elements are quadratic in x, y, and z, so that
    % Simpson's rule quadrature gives exact results.
    for ijk=1:nphase
        for k=1:3
            for j=1:3
                for i=1:3
                    x = (i-1)/2.0;
                    y = (j-1)/2.0;
                    z = (k-1)/2.0;

                    % dndx means the negative derivative, with respect to x, of the shape
                    % matrix N (see manual, Sec. 2.2), dndy, and dndz are similar.

                    dndx(1)=-(1.0-y)*(1.0-z);
                    dndx(2)=(1.0-y)*(1.0-z);
                    dndx(3)=y*(1.0-z);
                    dndx(4)=-y*(1.0-z);
                    dndx(5)=-(1.0-y)*z;
                    dndx(6)=(1.0-y)*z;
                    dndx(7)=y*z;
                    dndx(8)=-y*z;
                    dndy(1)=-(1.0-x)*(1.0-z);
                    dndy(2)=-x*(1.0-z);
                    dndy(3)=x*(1.0-z);
                    dndy(4)=(1.0-x)*(1.0-z);
                    dndy(5)=-(1.0-x)*z;
                    dndy(6)=-x*z;

                    dndy(7)=x*z;
                    dndy(8)=(1.0-x)*z;
                    dndz(1)=-(1.0-x)*(1.0-y);
                    dndz(2)=-x*(1.0-y);
                    dndz(3)=-x*y;
                    dndz(4)=-(1.0-x)*y;
                    dndz(5)=(1.0-x)*(1.0-y);
                    dndz(6)=x*(1.0-y);
                    dndz(7)=x*y;
                    dndz(8)=(1.0-x)*y;

                    % now build strain matrix
                    es = zeros(6, 8, 3);


                    for n=1:8
                        es(1,n,1)=dndx(n);
                        es(2,n,2)=dndy(n);
                        es(3,n,3)=dndz(n);
                        es(4,n,1)=dndz(n);
                        es(4,n,3)=dndx(n);
                        es(5,n,2)=dndz(n);
                        es(5,n,3)=dndy(n);
                        es(6,n,1)=dndy(n);
                        es(6,n,2)=dndx(n);
                    end

                    % Matrix multiply to determine value at (x,y,z), multiply by
                    % proper weight, and sum into dk, the stiffness matrix
                    for mm=1:3
                        for nn=1:3
                            for ii=1:8
                                for jj=1:8
                                % Define sum over strain matrices and elastic moduli matrix for
                                % stiffness matrix
                                sum = 0.0;
                                for kk=1:6
                                    for ll=1:6
                                        sum = sum+es(kk,ii,mm)*cmod(ijk,kk,ll)*es(ll,jj,nn);
                                    end
                                end    
                                dk(ijk,ii,mm,jj,nn)=dk(ijk,ii,mm,jj,nn)+g(i,j,k)*sum/216;
                                end
                            end
                        end
                    end    
                end
            end
        end
    end

    % Set up vector for linear term, b, and constant term, C,
    % in the elastic energy. This is done using the stiffness matrices,
    % and the periodic terms in the applied strain that come in at the
    % boundary pixels via the periodic boundary conditions and the
    % condition that an applied macroscopic strain exists (see Sec. 2.2
    % in the manual). It is easier to set b up this way than to analytically
    % write out all the terms involved.
    % Initialize b and C
    b = zeros(ns, 3);
    C = 0.0;

    %  For all cases, the correspondence between 1-8 finite element node
    %  labels and 1-27 neighbor labels is (see Table 4 in manual):
    %  1:ib(m,27), 2:ib(m,3),
    %  3:ib(m,2),4:ib(m,1),
    %  5:ib(m,26),6:ib(m,19)
    %  7:ib(m,18),8:ib(m,17)

    is = zeros(8, 1);
    is(1)=27;
    is(2)=3;
    is(3)=2;
    is(4)=1;
    is(5)=26;
    is(6)=19;
    is(7)=18;
    is(8)=17;

    %  x=nx face
    delta = zeros(8, 3);
    for i3=1:3
        for i8=1:8
            delta(i8,i3)=0.0;
            if (i8==2 | i8==3 | i8 == 6 | i8==7) 
                delta(i8,1)=exx*nx;
                delta(i8,2)=exy*nx;
                delta(i8,3)=exz*nx;
            end
        end
    end    
    for j=1:ny-1
        for k=1:nz-1
            m = nxy*(k-1)+j*nx;
            for nn=1:3
                for mm=1:8
                    sum = 0.0;
                    for m3=1:3
                        for m8=1:8
                            sum = sum+delta(m8,m3)*dk(pix(m),m8,m3,mm,nn);
                            C = C+0.5*delta(m8,m3)*dk(pix(m),m8,m3,mm,nn)*delta(mm,nn);
                        end
                    end     
                    b(ib(m,is(mm)),nn)=b(ib(m,is(mm)),nn)+sum;
                end
            end    
        end
    end    

    % y=ny face
    delta = zeros(8, 3);

    for i3=1:3
        for i8=1:8
            delta(i8,i3)=0.0;
            if(i8==3|i8==4|i8==7|i8==8) 
            delta(i8,1)=exy*ny;
            delta(i8,2)=eyy*ny;
            delta(i8,3)=eyz*ny;
            end 
        end
    end    
    for i=1:nx-1
        for k=1:nz-1
            m = nxy*(k-1)+nx*(ny-1)+i;
            for nn=1:3
                for mm=1:8
                    sum = 0.0;
                    for m3=1:3
                        for m8=1:8
                            sum=sum+delta(m8,m3)*dk(pix(m),m8,m3,mm,nn);
                            C=C+0.5*delta(m8,m3)*dk(pix(m),m8,m3,mm,nn)*delta(mm,nn);
                        end
                    end    
                    b(ib(m,is(mm)),nn)=b(ib(m,is(mm)),nn)+sum;
                end
            end    
        end
    end

    % z=nz face
    delta = zeros(8, 3);

    for i3=1:3
        for i8=1:8
            delta(i8,i3)=0.0;
            if(i8==5|i8==6|i8==7|i8==8) 
                delta(i8,1)=exz*nz;
                delta(i8,2)=eyz*nz;
                delta(i8,3)=ezz*nz;
            end
        end
    end    
    for i=1:nx-1
        for j=1:ny-1
            m = nxy*(nz-1)+nx*(j-1)+i;
            for nn=1:3
                for mm=1:8
                    sum=0.0;
                    for m3=1:3
                        for m8=1:8
                            sum=sum+delta(m8,m3)*dk(pix(m),m8,m3,mm,nn);
                            C=C+0.5*delta(m8,m3)*dk(pix(m),m8,m3,mm,nn)*delta(mm,nn);
                        end
                    end    
                    b(ib(m,is(mm)),nn)=b(ib(m,is(mm)),nn)+sum;
                end
            end    
        end
    end    

    %  x=nx y=ny edge
    delta = zeros(8, 3);

    for i3=1:3
        for i8=1:8
            delta(i8,i3)=0.0;
            if(i8==2||i8==6)
                delta(i8,1)=exx*nx;
                delta(i8,2)=exy*nx;
                delta(i8,3)=exz*nx;
            end 
            if(i8==4||i8==8)
                delta(i8,1)=exy*ny;
                delta(i8,2)=eyy*ny;
                delta(i8,3)=eyz*ny;
            end
            if(i8==3||i8==7) 
                delta(i8,1)=exy*ny+exx*nx;
                delta(i8,2)=eyy*ny+exy*nx;
                delta(i8,3)=eyz*ny+exz*nx;
            end 
        end
    end
    for k=1:nz-1
        m=nxy*k;
        for nn=1:3
            for mm=1:8
                sum=0.0;
                for m3=1:3
                    for m8=1:8
                        sum=sum+delta(m8,m3)*dk(pix(m),m8,m3,mm,nn);
                        C=C+0.5*delta(m8,m3)*dk(pix(m),m8,m3,mm,nn)*delta(mm,nn);
                    end
                end            
                b(ib(m,is(mm)),nn)=b(ib(m,is(mm)),nn)+sum;
            end
        end
    end

    % x=nx z=nz edge
    delta = zeros(8, 3);

    for i3=1:3
        for i8=1:8
            delta(i8,i3)=0.0;
            if(i8==2||i8==3) 
                delta(i8,1)=exx*nx;
                delta(i8,2)=exy*nx;
                delta(i8,3)=exz*nx;
            end
            if(i8==5||i8==8) 
                delta(i8,1)=exz*nz;
                delta(i8,2)=eyz*nz;
                delta(i8,3)=ezz*nz;
            end 
            if(i8==6||i8==7) 
                delta(i8,1)=exz*nz+exx*nx;
                delta(i8,2)=eyz*nz+exy*nx;
                delta(i8,3)=ezz*nz+exz*nx;
            end 
        end
    end
    for j=1:ny-1
        m=nxy*(nz-1)+nx*(j-1)+nx;
        for nn=1:3
            for mm=1:8
                sum=0.0;
                for m3=1:3
                    for m8=1:8
                        sum=sum+delta(m8,m3)*dk(pix(m),m8,m3,mm,nn);
                        C=C+0.5*delta(m8,m3)*dk(pix(m),m8,m3,mm,nn)*delta(mm,nn);
                    end
                end
                b(ib(m,is(mm)),nn)=b(ib(m,is(mm)),nn)+sum;
            end
        end    
    end

    % y=ny z=nz edge
    delta = zeros(8, 3);

    for i3=1:3
        for i8=1:8
            delta(i8,i3)=0.0;
            if(i8==5||i8==6) 
                delta(i8,1)=exz*nz;
                delta(i8,2)=eyz*nz;
                delta(i8,3)=ezz*nz;
            end 
            if(i8==3||i8==4) 
                delta(i8,1)=exy*ny;
                delta(i8,2)=eyy*ny;
                delta(i8,3)=eyz*ny;
            end 
            if(i8==7||i8==8) 
                delta(i8,1)=exy*ny+exz*nz;
                delta(i8,2)=eyy*ny+eyz*nz;
                delta(i8,3)=eyz*ny+ezz*nz;
            end
        end
    end

    for i=1:nx-1
        m=nxy*(nz-1)+nx*(ny-1)+i;
        for nn=1:3
            for mm=1:8
                sum=0.0;
                for m3=1:3
                    for m8=1:8
                        sum=sum+delta(m8,m3)*dk(pix(m),m8,m3,mm,nn);
                        C=C+0.5*delta(m8,m3)*dk(pix(m),m8,m3,mm,nn)*delta(mm,nn);
                    end
                end            
                b(ib(m,is(mm)),nn)=b(ib(m,is(mm)),nn)+sum;
            end
        end
    end

    % x=nx y=ny z=nz corner
    delta = zeros(8, 3);

    for i3=1:3
        for i8=1:8
            delta(i8,i3)=0.0;
            if(i8==2) 
                delta(i8,1)=exx*nx;
                delta(i8,2)=exy*nx;
                delta(i8,3)=exz*nx;
            end
            if(i8==4) 
                delta(i8,1)=exy*ny;
                delta(i8,2)=eyy*ny;
                delta(i8,3)=eyz*ny;
            end 
            if(i8==5) 
                delta(i8,1)=exz*nz;
                delta(i8,2)=eyz*nz;
                delta(i8,3)=ezz*nz;
            end 
            if(i8==8) 
                delta(i8,1)=exy*ny+exz*nz;
                delta(i8,2)=eyy*ny+eyz*nz;
                delta(i8,3)=eyz*ny+ezz*nz;
            end 
            if(i8==6) 
                delta(i8,1)=exx*nx+exz*nz;
                delta(i8,2)=exy*nx+eyz*nz;
                delta(i8,3)=exz*nx+ezz*nz;
            end 
            if(i8==3) 
                delta(i8,1)=exx*nx+exy*ny;
                delta(i8,2)=exy*nx+eyy*ny;
                delta(i8,3)=exz*nx+eyz*ny;
            end
            if(i8==7) 
                delta(i8,1) = exx*nx+exy*ny+exz*nz;
                delta(i8,2) = exy*nx+eyy*ny+eyz*nz;
                delta(i8,3) = exz*nx+eyz*ny+ezz*nz;
            end 
        end
    end
    m = nx*ny*nz;
    for nn=1:3
        for mm=1:8
            sum=0.0;
            for m3=1:3
                for m8=1:8
                    sum=sum+delta(m8,m3)*dk(pix(m),m8,m3,mm,nn);
                    C=C+0.5*delta(m8,m3)*dk(pix(m),m8,m3,mm,nn)*delta(mm,nn);
                end
            end
            b(ib(m,is(mm)),nn)=b(ib(m,is(mm)),nn)+sum;
        end
    end

end


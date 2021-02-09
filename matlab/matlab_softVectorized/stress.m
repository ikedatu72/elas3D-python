function [strxx,stryy,strzz,strxz,stryz,strxy,sxx,syy,szz,sxz,syz,sxy] = stress(nx,ny,nz,ns,exx,eyy,ezz,exz,eyz,exy,u,ib,pix,cmod)
%Subroutine that computes the six average stresses and six

nxy=nx*ny;
% set up single element strain matrix
% dndx, dndy, and dndz are the components of the average strain
% matrix in a pixel
dndx = zeros(8, 1);
dndy = zeros(8, 1);
dndz = zeros(8, 1);

dndx(1)=-0.25;
dndx(2)=0.25;
dndx(3)=0.25;
dndx(4)=-0.25;
dndx(5)=-0.25;
dndx(6)=0.25;
dndx(7)=0.25;
dndx(8)=-0.25;
dndy(1)=-0.25;
dndy(2)=-0.25;
dndy(3)=0.25;
dndy(4)=0.25;
dndy(5)=-0.25;
dndy(6)=-0.25;
dndy(7)=0.25;
dndy(8)=0.25;
dndz(1)=-0.25;
dndz(2)=-0.25;
dndz(3)=-0.25;
dndz(4)=-0.25;
dndz(5)=0.25;
dndz(6)=0.25;
dndz(7)=0.25;
dndz(8)=0.25;

% Build averaged strain matrix, follows code in femat, but for average
% strain over the pixel, not the strain at a point.
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

% Compute components of the average stress and strain tensors in each pixel
strxx=0.0;
stryy=0.0;
strzz=0.0;
strxz=0.0;
stryz=0.0;
strxy=0.0;
sxx=0.0;
syy=0.0;
szz=0.0;
sxz=0.0;
syz=0.0;
sxy=0.0;
for k=1:nz
    for j=1:ny
        for i=1:nx
            m=(k-1)*nxy+(j-1)*nx+i;
            % load in elements of 8-vector using pd. bd. conds.
            for mm=1:3
                uu(1,mm)=u(m,mm);
                uu(2,mm)=u(ib(m,3),mm);
                uu(3,mm)=u(ib(m,2),mm);
                uu(4,mm)=u(ib(m,1),mm);
                uu(5,mm)=u(ib(m,26),mm);
                uu(6,mm)=u(ib(m,19),mm);
                uu(7,mm)=u(ib(m,18),mm);
                uu(8,mm)=u(ib(m,17),mm);
            end
            % Correct for periodic boundary conditions, some displacements are wrong
            % for a pixel on a periodic boundary. Since they come from an opposite
            % face, need to put in applied strain to correct them.
            if(i==nx) 
                uu(2,1)=uu(2,1)+exx*nx;
                uu(2,2)=uu(2,2)+exy*nx;
                uu(2,3)=uu(2,3)+exz*nx;
                uu(3,1)=uu(3,1)+exx*nx;
                uu(3,2)=uu(3,2)+exy*nx;
                uu(3,3)=uu(3,3)+exz*nx;
                uu(6,1)=uu(6,1)+exx*nx;
                uu(6,2)=uu(6,2)+exy*nx;
                uu(6,3)=uu(6,3)+exz*nx;
                uu(7,1)=uu(7,1)+exx*nx;
                uu(7,2)=uu(7,2)+exy*nx;
                uu(7,3)=uu(7,3)+exz*nx;
            end 
            if(j==ny) 
                uu(3,1)=uu(3,1)+exy*ny;
                uu(3,2)=uu(3,2)+eyy*ny;
                uu(3,3)=uu(3,3)+eyz*ny;
                uu(4,1)=uu(4,1)+exy*ny;
                uu(4,2)=uu(4,2)+eyy*ny;
                uu(4,3)=uu(4,3)+eyz*ny;
                uu(7,1)=uu(7,1)+exy*ny;
                uu(7,2)=uu(7,2)+eyy*ny;
                uu(7,3)=uu(7,3)+eyz*ny;
                uu(8,1)=uu(8,1)+exy*ny;
                uu(8,2)=uu(8,2)+eyy*ny;
                uu(8,3)=uu(8,3)+eyz*ny;
            end
            if(k==nz) 
                uu(5,1)=uu(5,1)+exz*nz;
                uu(5,2)=uu(5,2)+eyz*nz;
                uu(5,3)=uu(5,3)+ezz*nz;
                uu(6,1)=uu(6,1)+exz*nz;
                uu(6,2)=uu(6,2)+eyz*nz;
                uu(6,3)=uu(6,3)+ezz*nz;
                uu(7,1)=uu(7,1)+exz*nz;
                uu(7,2)=uu(7,2)+eyz*nz;
                uu(7,3)=uu(7,3)+ezz*nz;
                uu(8,1)=uu(8,1)+exz*nz;
                uu(8,2)=uu(8,2)+eyz*nz;
                uu(8,3)=uu(8,3)+ezz*nz;
            end
            % local stresses and strains in a pixel
            s11 = sum(squeeze(es(1, :, :)).* uu, 'all');
            s22 = sum(squeeze(es(2, :, :)).* uu, 'all');
            s33 = sum(squeeze(es(3, :, :)).* uu, 'all');
            s13 = sum(squeeze(es(4, :, :)).* uu, 'all');
            s23 = sum(squeeze(es(5, :, :)).* uu, 'all');
            s12 = sum(squeeze(es(6, :, :)).* uu, 'all');
            
            str11 = 0;
            str22 = 0;
            str33 = 0;
            str13 = 0;
            str23 = 0;
            str12 = 0;
            
            for n=1:6
                str11=str11+sum(cmod(pix(m),1,n)*squeeze(es(n,:,:)).*uu,'all');
                str22=str22+sum(cmod(pix(m),2,n)*squeeze(es(n,:,:)).*uu,'all');
                str33=str33+sum(cmod(pix(m),3,n)*squeeze(es(n,:,:)).*uu,'all');
                str13=str13+sum(cmod(pix(m),4,n)*squeeze(es(n,:,:)).*uu,'all');
                str23=str23+sum(cmod(pix(m),5,n)*squeeze(es(n,:,:)).*uu,'all');
                str12=str12+sum(cmod(pix(m),6,n)*squeeze(es(n,:,:)).*uu,'all');
            end

            % sum local strains and stresses into global values
            strxx=strxx+str11;
            stryy=stryy+str22;
            strzz=strzz+str33;
            strxz=strxz+str13;
            stryz=stryz+str23;
            strxy=strxy+str12;
            sxx=sxx+s11;
            syy=syy+s22;
            szz=szz+s33;
            sxz=sxz+s13;
            syz=syz+s23;
            sxy=sxy+s12;
        end
    end
end

% Volume average of global stresses and strains
strxx=strxx/ns;
stryy=stryy/ns;
strzz=strzz/ns;
strxz=strxz/ns;
stryz=stryz/ns;
strxy=strxy/ns;

sxx=sxx/ns;
syy=syy/ns;
szz=szz/ns;
sxz=sxz/ns;
syz=syz/ns;
sxy=sxy/ns;

end



function [pix] = ppixel(nx, ny, nz, ns, nphase)
    %  Subroutine that sets up microstructural image
    pix = zeros(ns, 1);
    
    %  (USER)  If you want to set up a test image inside the program, instead of
    %  reading it in from a file, this should be done inside this subroutine.
    f = fopen("microstructure.dat", "r");
    
    nxy = nx*ny;
    
    for k = 1:nz
        for j = 1:ny
            for i = 1:nx
                m = nxy*(k-1)+nx*(j-1)+i;
                pix(m) = str2double(fgetl(f));
            end
        end
    end
    
    fclose(f);
    
    %  Check for wrong phase labels--less than 1 or greater than nphase
    %  FOR PYTHON, label starts from 0. So we check for any label which is less than 0
       
    for m = 1:ns
        if pix(m) < 0
            display("Phase label in pix < 1--error at " + m); 
        end
        
        if pix(m) > nphase
            print("Phase label in pix > nphase--error at " + m); 
        end
    
    end
end


function [prob] = assig(ns, nphase, pix)

    % Subroutine that counts volume fractions
    prob = zeros(nphase, 1);

    for m=1:ns
        for i=1:nphase
            if pix(m) == i
                prob(i) = prob(i)+1;
            end
        end
    end

    for i=1:nphase
        prob(i)=prob(i)/ns;
    end

end


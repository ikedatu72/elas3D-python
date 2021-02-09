function [prob] = assig(ns, nphase, pix)

    % Subroutine that counts volume fractions
    prob = zeros(nphase, 1);

    for i=1:nphase
        prob(i) = sum(pix == i)/ns;
    end


end


function [x_proj] = Proj(x, U, P, l)

    x_proj = zeros(P,1);
    
    for j=1:l
        sca = U(:,j).'*x;
        x_proj = x_proj + sca*U(:,j);
    end
    
    x_proj = x_proj;
end
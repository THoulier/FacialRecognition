function [K] = crit_K(l,vp,N)
    denum = 0;
    num = 0;
    
    for i=1:N
        denum = denum + vp(i);
    end
    
    for i=1:l
        num = num + vp(i);
    end
    
    K = (num/denum);
end
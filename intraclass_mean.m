function [mean_class] = intraclass_mean(data_trn, N, cls_trn, U, m_empirique, min_l)
    %return une matrice 6x8, chaque ligne est une moyenne de classe
    mean_class = [];

    for i=1:(N/size(cls_trn,1)):N
        S = 0;
        
        w = [];
        
        if i>50
            len = (N/size(cls_trn,1))-1;
        else
            len = (N/size(cls_trn,1));
        end
        
        
        for j=1:len
            weight_xj = [];
            for k=1:min_l
                weight_xj = [weight_xj U(:,k).'*(data_trn(:,j+i-1) - m_empirique)];  
            end
            w = [w ; weight_xj]
        end
        S = sum(w);
        
        mean_class = [mean_class ; S/size(cls_trn,1)];
    end


end
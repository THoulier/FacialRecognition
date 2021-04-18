function [w] = weigth(data_trn_class, m_empirique, min_l, U)

w = [];

for i=1:size(data_trn_class,2)
    w_x = [];
    for j=1:min_l
       w_x = [w_x  U(:,j).'*(data_trn_class(:,i) - m_empirique)];
    end
    w = [w ; w_x];
end

end
function [k_neighbors_index] = knn_classifier(x, data_trn, m_empirique, N, U, min_l, k)

    weight_x = [];
    list_w = [];
    V = [];

    for j=1:min_l
        weight_x = [weight_x  U(:,j).'*(x - m_empirique)];
    end
    
    
    for i=1:N
        weight_ij = [];
        for j=1:min_l
            weight_ij = [weight_ij  U(:,j).'*(data_trn(:,i) - m_empirique)];
        end
        list_w = [list_w ; weight_ij];
    end
    
    %comparaison image d'entree avec toutes les images de la base
    for i=1:N
        V = [V norm(weight_x - list_w(i,:), 2)];
    end
    
    k_neighbors_values = zeros(1,k);
    k_neighbors_index = zeros(1,k);
    V_bis = V;
        
    for i=1:k
        mini = 0;
        [k_neighbors_value(i),mini] = min(V);
        k_neighbors_index(i) = find(V_bis == k_neighbors_value(i));
        V = [V(1:mini-1) V(mini+1:end)];
    end
    
end
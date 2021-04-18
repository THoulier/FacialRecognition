function [class] = find_class(k_index,k, class_list)
    class = [];
    
    for i=1:k
        if k_index(i) >= 1 && k_index(i) <= 10
            class(i) = 1;
        elseif k_index(i) > 10 && k_index(i) <= 20
            class(i) = 5;
        elseif k_index(i) > 20 && k_index(i) <= 30
            class(i) = 9;
        elseif k_index(i) > 30 && k_index(i) <= 40
            class(i) = 22;
        elseif k_index(i) > 40 && k_index(i) <= 50
            class(i) = 26;
        elseif k_index(i) > 50 && k_index(i) <= 60
            class(i) = 28;
        end
    end
    
    [value, index] = max(histc(class, class_list));
    class = class_list(index);
    
end
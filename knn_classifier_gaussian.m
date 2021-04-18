function [Sigma, f_w, class_predict] = knn_classifier_gaussian(x, data_trn,mu, m, N, m_empirique, min_l, U, cls_test)
Sigma = zeros(min_l);
w = [];
for i=1:size(data_trn,2)/m:N
   w = [w ; weigth(data_trn(:,i:i+10-1), m_empirique, min_l, U)];
end

%determination matrice de covariance
cpt=1;
for i=1:m
    Sigma = Sigma + sigma(i,w(cpt:cpt-1+size(data_trn,2)/m,:),mu,min_l);
    cpt=cpt+10;
end

% for i=1:m
%     for j=1:N
%         Sigma = Sigma + (w(j,:) - mu(i,:))*((w(j,:) - mu(i,:)).');
%     end
% end

Sigma = (1/N).*Sigma;

w_x = [];

for j=1:min_l
   w_x = [w_x  U(:,j).'*(x - m_empirique)];
end

%Maximum de vraissemblance
f_w = [];

for i=1:m
    f_w = [f_w  sqrt(1/(2*pi*det(Sigma)))*exp(-(1/2)*((w_x-mu(i,:))*(Sigma^(-1))*((w_x-mu(i,:)).')))];
end

[maxi,index] = max(f_w);
class_predict = cls_test(index);
end

function [Sigma] = sigma(i,w,mu, min_l)
Sigma = zeros(min_l);

for j=1:10
   Sigma = Sigma + ((w(j,:) - mu(i,:)).')*(w(j,:) - mu(i,:));
end


end
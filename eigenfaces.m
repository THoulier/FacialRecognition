% P. Vallet (Bordeaux INP), 2019

clc;
clear all;
close all;

%% Data extraction

% Training set
adr = './database/training1/';
fld = dir(adr);
nb_elt = length(fld);



% Data matrix containing the training images in its columns 
data_trn = []; 

% Vector containing the class of each training image
lb_trn = []; 

for i=1:nb_elt
    if fld(i).isdir == false
        lb_trn = [lb_trn ; str2num(fld(i).name(6:7))];
        img = double(imread([adr fld(i).name]));
        data_trn = [data_trn img(:)];
    end
end

% Size of the training set
[P,N] = size(data_trn);

% Classes contained in the training set
[~,I]=sort(lb_trn);
data_trn = data_trn(:,I);
[cls_trn,bd,~] = unique(lb_trn);
Nc = length(cls_trn); 

% Number of training images in each class
size_cls_trn = [bd(2:Nc)-bd(1:Nc-1);N-bd(Nc)+1]; 

% Display the database
F = zeros(192*Nc,168*max(size_cls_trn));

for i=1:Nc
    for j=1:size_cls_trn(i)
          pos = sum(size_cls_trn(1:i-1))+j;
          F(192*(i-1)+1:192*i,168*(j-1)+1:168*j) = reshape(data_trn(:,pos),[192,168]);
    end
end

figure;
imagesc(F);
colormap(gray);
axis off;



%% Covariance matrix

m_empirique = zeros(P,1);
for i=1:N
    m_empirique = m_empirique + data_trn(:,i);
end

m_empirique = (1/N)*m_empirique;
X = (1/sqrt(N))*(data_trn - m_empirique);
%R = X*X.';

A = X.'*X;
[V,vp]=eig(A, 'vector');

[vp, index] = sort(vp, 'descend');
V = V(:,index);
%V = V(:,2:end);


U = real((X*V)*((V.'*X.'*X*V)^(-1/2)));




%% Eigenfaces display

figure;
for i=1:N-1
    eignfaces = reshape(U(:,i),[192,168]);
    subplot(6,10,i);
    imagesc(eignfaces);
    colormap(gray);
end

%% Orthogonal projection

x_proj = Proj(X(:,11), U, P, 60);
x_init = reshape(X(:,11),[192,168]);

test = reshape(x_proj,[192,168]);

%Image de base vs image projetee
figure;
subplot(121)
imagesc(x_init);
colormap(gray);
subplot(122)
imagesc(test);
colormap(gray);

%Evolution image reconstruite en fonction de l
figure;
for l=1:N-1
    x_proj = Proj(data_trn(:,1), U, P, l);
    x_proj = reshape(x_proj,[192,168]);
    subplot(6,10,l);
    imagesc(x_proj);
    colormap(gray);
end

%% Dimension reduction


%kappa criteria use to determine optimal l value
alpha = 0.9;
min_l=N;
K = [];

for l=1:N-1
    K = [K crit_K(l,vp,N-1)];
    if  K(l) >= alpha
        if l < min_l
            min_l = l;
        end
    end
end

figure;
plot(K)

%% Classifieur k-NN classique

% Import test vectors
adr = './database/test5/';
fld = dir(adr);
nb_elt = length(fld);

data_test = []; 
lb_test = []; 

for i=1:nb_elt
    if fld(i).isdir == false
        img = double(imread([adr fld(i).name]));
        lb_test = [lb_test ; str2num(fld(i).name(6:7))];
        data_test = [data_test img(:)];
    end
end

%print faces from data_test

[P_test,N_test] = size(data_test);

[~,I_test]=sort(lb_test);
data_test = data_test(:,I_test);
[cls_test,bd_test,~] = unique(lb_test);
Nc_test = length(cls_test); 

size_cls_test = [bd_test(2:Nc_test)-bd_test(1:Nc_test-1);N_test-bd_test(Nc_test)+1]; 

F_test = zeros(192*Nc_test,168*max(size_cls_test));

for i=1:Nc_test
    for j=1:size_cls_test(i)
          pos_test = sum(size_cls_test(1:i-1))+j;
          F_test(192*(i-1)+1:192*i,168*(j-1)+1:168*j) = reshape(data_test(:,pos_test),[192,168]);
    end
end

figure;
imagesc(F_test);
colormap(gray);
axis off;

%k-NN

k=5;
k_index_mat = [];


for i=1:size(data_test,2)
    k_index_mat = [k_index_mat ; find_class(knn_classifier(data_test(:,i), data_trn, m_empirique, N, U, min_l,k),k, cls_test)];
end

%labels predicted
lb_predict = k_index_mat;

[Conf_Mat, Err_Rate] = confmat(lb_test,lb_predict);

Conf_Mat
Err_Rate

%% Classifieur k-NN Gaussien

%Moyenne intra classe
mu = intraclass_mean(data_trn, N, cls_trn, U, m_empirique, min_l);

%Variance intra classe

%Affichage couple de composantes principales
w = [];
for i=1:10:30
   w = [w ; weigth(data_trn(:,i:i+10-1), m_empirique, min_l, U)];
end

figure;
subplot(221)
scatter(w(1:10,1), w(1:10,2),10,'MarkerEdgeColor',[0 .5 .5]);
hold on;
scatter(w(10:20,1), w(10:20,2),10,'MarkerEdgeColor',[1 0 0]);
hold on;
scatter(w(20:30,1), w(20:30,2),10,'MarkerEdgeColor',[0 1 0]);
title("(1,2)");
subplot(222)
scatter(w(1:10,2), w(1:10,3),10,'MarkerEdgeColor',[0 .5 .5]);
hold on;
scatter(w(10:20,2), w(10:20,3),10,'MarkerEdgeColor',[1 0 0]);
hold on;
scatter(w(20:30,2), w(20:30,3),10,'MarkerEdgeColor',[0 1 0]);
title("(2,3)")
subplot(223)
scatter(w(1:10,3), w(1:10,4),10,'MarkerEdgeColor',[0 .5 .5]);
hold on;
scatter(w(10:20,3), w(10:20,4),10,'MarkerEdgeColor',[1 0 0]);
hold on;
scatter(w(20:30,3), w(20:30,4),10,'MarkerEdgeColor',[0 1 0]);
title("(3,4)")
subplot(224)
scatter(w(1:10,4), w(1:10,5),10,'MarkerEdgeColor',[0 .5 .5]);
hold on;
scatter(w(10:20,4), w(10:20,5),10,'MarkerEdgeColor',[1 0 0]);
hold on;
scatter(w(20:30,4), w(20:30,5),10,'MarkerEdgeColor',[0 1 0]);
title("(4,5)")

%Gaussian classifier

lb_predict_gauss=[];
for i=1:size(data_test,2)
    [Sigma, f, class] = knn_classifier_gaussian(data_test(:,i), data_trn, mu, 6, N, m_empirique, min_l, U, cls_test);
    lb_predict_gauss = [lb_predict_gauss ;class];
end

[Conf_Mat_gauss, Err_Rate_gauss] = confmat(lb_test,lb_predict_gauss)


Conf_Mat_gauss
Err_Rate_gauss



%% Classifier comparison

%Getting the values by running the program for each test base
perf_classcial = [0,0, 0.1061,0.4444,0.6778,0.7917];
perf_gauss = [0,0,0.0303,0.2222,0.8111,0.8333]
test_number = [1,2,3,4,5,6];

%Display performance score for each K-NN classifier
plot(test_number,perf_classcial);
hold on;
plot(test_number,perf_gauss);
grid on;
legend("Classical k-NN", "Gaussian k-NN")
xlabel("Test base number (1 to 6)")
ylabel("Error rate")


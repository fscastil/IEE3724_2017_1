load('training');
load('testing');

[Xz, mux, sigmax] = zscore(X);
Xtz = (Xt - mux) ./ sigmax;

step = 302;
F = [];
Ft = [];

for i=1:step:4831
    disp([num2str(i) ' hasta ' num2str(i + step - 1)])
    S = Xz(:,i:i+step-1);
    S = S(:, [1:59 60+36:end]);
    clean = Bfs_clean(S, d);
    S = S(:,clean);
    [T,U,P,Q,W,B] = Bft_plsr(S, d, 60);
    [Tz, mut, sigmat] = zscore(T);
    F = [F Tz];

    St = Xtz(:,i:i+step-1);
    St = St(:, [1:59 60+36:end]);
    St = St(:,clean);
    Tt = St * W;
    Ttz = (Tt - mut) ./ sigmat;
    Ft = [Ft Tt];
end

% OBTENER PCA
% [Y,lambda,A,Xs,mx] = Bft_pca(F, 0.99);
% [Yz, muy, sigmay] = zscore(Y);
%
%
%
% [P,U,~,Q,W,B] = Bft_plsr(Yz, d, 60);
% P = double(P);
% [Pz, mup, sigmap] = zscore(P);


[Fz, muf, sigmaf] = zscore(F);
[T,U,P,Q,W,B] = Bft_plsr(Fz, d, 60);
T = double(T);
[Tz, mut, sigmat] = zscore(T);

Tt = double(((Ft - muf) ./ sigmaf) * W);
Ttz = double((Tt - mut) ./ sigmat);


% B = A(:,1:807);
% Mxt = ones(2744, 1)*mx;
% X0t = Ft - Mxt;
% Yt = X0t * B;
% Ytz = (Yt - muy(1:807)) ./ sigmay(1:807);
%
% % REALIZAR MISMO PSLR SOBRE TESTING
% Pt = double(((Ytz - muy) ./ sigmay) * W);
% Ptz = double((Pt - mup) ./ sigmap);


clear b
k = 0;
k=k+1;b(k).name = 'lda';   b(k).options.p = [];         b(k).string = 'LDA';            % LDA
% k=k+1;b(k).name = 'qda';   b(k).options.p = [];         b(k).string = 'QDA';            % QDA
k=k+1;b(k).name = 'dmin';  b(k).options = [];           b(k).string = 'Euclidean';      % Euclidean distance
% k=k+1;b(k).name = 'knn';   b(k).options.k = 3;           b(k).string = 'KNN3';      % Euclidean distance
% k=k+1;b(k).name = 'knn';   b(k).options.k = 5;           b(k).string = 'KNN5';      % Euclidean distance
% k=k+1;b(k).name = 'knn';   b(k).options.k = 7;           b(k).string = 'KNN7';      % Euclidean distance
% k=k+1;b(k).name = 'knn';   b(k).options.k = 9;           b(k).string = 'KNN9';      % Euclidean distance
% k=k+1;b(k).name = 'knn';   b(k).options.k = 11;           b(k).string = 'KNN11';      % KNN
% k=k+1;b(k).name = 'knn';   b(k).options.k = 13;           b(k).string = 'KNN13';      % KNN
% k=k+1;b(k).name = 'knn';   b(k).options.k = 15;           b(k).string = 'KNN15';      % KNN
% k=k+1;b(k).name = 'knn';   b(k).options.k = 17;           b(k).string = 'KNN17';      % KNN
% k=k+1;b(k).name = 'knn';   b(k).options.k = 19;           b(k).string = 'KNN19';      % KNN
% k=k+1;b(k).name = 'knn';   b(k).options.k = 21;           b(k).string = 'KNN21';      % KNN
% k=k+1;b(k).name = 'knn';   b(k).options.k = 23;           b(k).string = 'KNN23';      % KNN
% k=k+1;b(k).name = 'knn';   b(k).options.k = 25;           b(k).string = 'KNN25';      % KNN
k=k+1;b(k).name = 'libsvm';   b(k).options.kernel = '-t 0';    b(k).options.Display='off';       b(k).string = 'SVMLIN';      % Euclidean distance
k=k+1;b(k).name = 'libsvm';   b(k).options.kernel = '-t 1';    b(k).options.Display='off';       b(k).string = 'SVMQUAD';      % Euclidean distance
k=k+1;b(k).name = 'libsvm';   b(k).options.kernel = '-t 2';    b(k).options.Display='off';       b(k).string = 'SVMPOLY';      % Euclidean distance
k=k+1;b(k).name = 'libsvm';   b(k).options.kernel = '-t 3';    b(k).options.Display='off';       b(k).string = 'SVMRBF';      % Euclidean distance

opc = b;
ds = Bcl_structure(Tz,double(d),Ttz,opc);                                   % ds has k columns

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 7. Evaluation of Performance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p = Bev_performance(ds,dt);

% 6. Output
for i=1:length(b)
    fprintf('%15s = %6.2f%%\n',b(i).string,p(i)*100);
end

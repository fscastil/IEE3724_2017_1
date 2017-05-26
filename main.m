% IMPORTANTE IMAGENES DEBEN ESTAR EN
% LA DIRECCION/CARPETA ./faces/
% RELATIVO A ESTE ARCHIVO (faces_lh en realidad)

% EXTRER CARACTERISTICAS DE LAS IMAGENES
Orig = faces_lh(51,120,1,10);
Origt = faces_lh(51,120, 11,11);

% LOS LABELS COMIENZAN DESDE 50
d = Bds_labels(10*ones(143,1));
dt = Bds_labels(1*ones(143,1));

% REMOVER LBP INVARIANTE A LA ROTACION
X = Orig(:, [1:944 1521:3760]);
Xt =  Origt(:, [1:944 1521:3760]);

% NORMALIZACION
[Xz, muzx, sigmazx] = zscore(X);
% EL TESTING DEBE SER NORMALIZADO BAJO LOS MISMOS
% PARAMETROS QUE EL TRAINING
Xtz = (Xt - muzx) ./ sigmazx;

% LIMPIEZA DE VARIABLES
clean = Bfs_clean(Xz, d);
Xz = Xz(:, clean);
Xtz = Xtz(:, clean);

% REALIZAR SFS
op.m = 26;
op.show = 1;
op.b.name = 'fisher';
s = Bfs_sfs(Xz, double(d), op);
s = [1043	1023	1035	1037	1437	289	1011	1319	1249	1464	516	1281	279	547	639	575	446	92	551	274	996	3	1277	1248	1235	1202	461	1478	241	741];

% OBTENER PCA
[Y,lambda,A,Xs,mx] = Bft_pca(Xz, 0.99);
[Yz, muy, sigmay] = zscore(Y);

% OBTENER PSLR
% EL - 50 ES POR QUE SI NO NO FUNCIONA EL PSLR
[T,U,P,Q,W,B] = Bft_plsr(Y, d, 49);
[Tz, mut, sigmat] = zscore(T);

% F ES LA MATRIZ DE TRAINING FINAL
F = [X(:,s(1:26)) T];
[Fz, muf, sigmaf] = zscore(F);

% REALIZAR MISMA TRANSFORMACION DE PCA SOBRE TESTING
% 546 ES EL TAMAÑO DE Y
B = A(:,1:814);
Mxt = ones(143, 1)*mx;
X0t = Xtz - Mxt;
Yt = X0t * B;
Ytz = (Yt - muy(1:814)) ./ sigmay(1:814);

% REALIZAR MISMO PSLR SOBRE TESTING
Tt = ((Yt - muy) ./ sigmay) * W;
Ft = [Xt(:,s(1:26)) Tt];

% COMBINAR SFS Y PSLR
Ftz = (Ft - muf) ./ sigmaf;

opknn.k = 1;
ds = Bcl_knn(Fz, d, Ftz, opknn);
p_normf = Bev_performance(ds,dt)

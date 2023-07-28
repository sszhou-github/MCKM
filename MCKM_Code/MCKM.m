clear
clc
folder='data';
addpath(genpath(folder));
addpath(genpath(pwd))

load('D1.mat')
% data normalization
X=data;
X = bsxfun(@rdivide,bsxfun(@minus, X, min(X, [], 1)), max(X, [], 1) - min(X, [], 1));
[m,n]=size(X);

%% Parameters
rho=1;
k_n =3;
phi = 0.9;
gamma =1;

%% MPS
Rand=randperm(size(X,1),1);
center0=X(Rand,:);
min_impro=1/(rho*(size(X,1)*size(X,2)).^0.5);
tic
max_it=10000;
[intial_center,K]=MPS(X,min_impro);%%% min_impro                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
if K>size(X,1)
    K=K-1;
end
[index_final,new_center,sumD,D]=kmeans(X,K,'Start',intial_center);

%% CM
dataMatrix=new_center';
[dim.d,dim.n] = size(dataMatrix);

%% Compute weights
[weightVec1,NodeArcMatrix] = compute_weight(dataMatrix,k_n,phi);

%% Construct Amap
A0 = NodeArcMatrix;
Ainput.A = A0;
Ainput.Amap = @(x) x*A0;
Ainput.ATmap = @(x) x*A0';
Ainput.ATAmat = A0*A0'; %%graph Laplacian
Ainput.ATAmap = @(x) x*Ainput.ATAmat;
dim.E = length(weightVec1);
options.stoptol = 1e-6; %% tolerance for terminating the algorithm
options.num_k = k_n; %%number of nearest neighbors
%%==============================================
%% Stopping Criteria 
%% use_kkt == 1: based on relative KKT residual
%% use_kkt == 0: based on relative duality gap
%% Note that we will always terminate fast AMA based on relative duality gap
%%==============================================
options.use_kkt = 0;
%%===============================================
%% Implemented Algorithms
%% run_fastama == 1 : run fast AMA 
%% Reference: Eric C. Chi & Kenneth Lange, Splitting methods for convex clustering, JCGS 2015
%% run_admm == 1: run ADMM (used as Phase I of SSNAL if warmstart applied)
%% run_ssnal == 1: run SSNAL
run_fastama = 0;
run_admm  = 1;
run_ssnal = 0;

%% run_fastama == 1 : run fast AMA 
if (run_fastama == 1)
    weightVec = gamma*weightVec1;
    %% max iteration number for fast AMA
    options.maxiter = 1000;
    [~, X_fastAMA] =fast_AMA(Ainput,dataMatrix,dim,weightVec,options);
    merge_result=X_fastAMA;
    tolClustering = options.stoptol;
    [multi_prototypes_cluster_id, K_ture] = find_cluster(merge_result,tolClustering);
end


%% run_admm == 1: run ADMM
if (run_admm == 1)
    options.use_kkt = 1;
    weightVec = gamma*weightVec1;
    %% max iteration number for ADMM
    options.maxiter = 100;
    [~,~,X_ADMM] =ADMM(Ainput,dataMatrix,dim,weightVec,options);
    merge_result=X_ADMM;
    tolClustering = options.stoptol;
    [multi_prototypes_cluster_id, K_ture] = find_cluster(X_ADMM,tolClustering);
end

%% run_ssnal == 1: run SSNAL
if (run_ssnal == 1)
    options.use_kkt = 1;
    weightVec = gamma*weightVec1;
    %% max iteration number for SSNAL
    options.maxiter = 100;
    %% Warmstart SSNAL with ADMM (optional)
    options.admm_iter = 50;
    [~,~,X_SSNAL] =SSNAL(Ainput,dataMatrix,dim,weightVec,options);
    merge_result=X_SSNAL;
    tolClustering = options.stoptol;
    [multi_prototypes_cluster_id, K_ture] = find_cluster(merge_result,tolClustering);
end
time=toc;
[global_label_final,Z]=final_samples_label(multi_prototypes_cluster_id,index_final,K);
cluster_result_convex=merge_result';
[FM1, ARI1, NMI1] = evaluate(global_label_final, label);

%%% visualization of MPS
% figure(1)
% for i=1:size(X,1)
%     plot(X(index_final==i,1),X(index_final==i,2),'.','MarkerSize',12);
%     hold on
% %     voronoi(new_center(:,1),new_center(:,2),new_center(:,3),'k--')
% %     h1=plot(new_center(:,1),new_center(:,2),'pk','MarkerSize',15,'MarkerFaceColor','k');
% end
% h1=plot(new_center(:,1),new_center(:,2),'pk','MarkerSize',15,'MarkerFaceColor','k');
% hold on
% 
% %%%%%2D voronoi
% hold on
% voronoi(new_center(:,1),new_center(:,2),"k--")

%%% visualization of MPS
% gplot(Z,[new_center(:,1),new_center(:,2)],'g-')
% hold on
% plot(new_center(:,1),new_center(:,2),'pk','MarkerSize',15,'MarkerFaceColor','k');
% hold on
% title('$$q=3$$','interpreter','latex')
% hold on
% xlim([0,1])
% ylim([0,1])
% set(gca,'LineWidth',1.2)





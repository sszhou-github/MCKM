function [weightVec,NodeArcMatrix,weightMatrix] = compute_weight(X,k,phi)
    tstart = clock;
    [k_nearest,dist] = knnsearch(X',X','K',k+1);
    len = size(X,2);
    %% construct weight
    weightMatrix = sparse(len,len);
    for i = 1:len
        for j=1:k+1
            weightMatrix(i,k_nearest(i,j)) = exp(-phi*dist(i,j)^2);
            weightMatrix(k_nearest(i,j),i) = exp(-phi*dist(i,j)^2);
        end
        weightMatrix(i,i) = 0;
    end
    %%construct W, Wbar
    [idx_r,idx_c,val] = find(triu(full(weightMatrix)));
    %[idx_r,idx_c,val] = find(full(weightMatrix));
    num_weight = length(val);
    W = sparse(len,num_weight);
    Wbar = sparse(len,num_weight);
    for i = 1:1:num_weight
        W(idx_r(i),i) = 1;
        Wbar(idx_c(i),i) = 1;
    end
    weightVec = val';
    NodeArcMatrix = W-Wbar;
    fprintf('\ntime taken to generate weight matrix = %3.2f\n',etime(clock,tstart));
%%********************************************************************    
    
    
    
    
    
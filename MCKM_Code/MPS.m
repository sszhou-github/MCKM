function [data1,K]=MPS(X,min_impro)
center0=sum(X)./size(X,1);
out0 =(dist2fcm(center0, X).^2)';
[dist0,~]=min(out0,[],2);
V0=sum(dist0);

data=center0;
out = (dist2fcm(data, X).^2)';
[dist,~]=min(out,[],2);
V1(1)=1;
K=2;
dist2=dist;
data1=data;
V=sum(dist);
while K<=size(X,1)
    [~,add_index]=max(dist2);
    data1=[data1;X(add_index,:)];
    out3 = (dist2fcm(data1, X).^2)';
    [dist3,~]=min(out3,[],2);
    V1(K)=sum(dist3);
    if norm(((V-V1(K)))/V)<=min_impro  break; end,
    V=V1(K);
    dist2=dist3;
    K=K+1;
end
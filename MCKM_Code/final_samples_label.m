
function[global_label_final,Z1]=final_samples_label(cluster_id,index_final,K)
for i=1:K
    for j=1:K
        if cluster_id(i)==cluster_id(j) 
            Z1(i,j)=1;
        else
            Z1(i,j)=0;
        end
    end
end
C = adj2cluster(Z1);
label_final=zeros(size(index_final,1),1);
for i=1:length(C)
    dd=cell2mat(C(i));
    for j=1:length(dd)
       label_final(index_final==dd(j))=i;
    end
end
global_label_final=label_final;
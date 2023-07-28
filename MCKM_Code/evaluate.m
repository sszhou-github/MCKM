function [FM, ARI, NMI,Accuracy] = evaluate(label1,label2)
%label1 is the true label and label2 is the predect label.
N = length(label2);% ��������
p = unique(label1);%���ص��Ǻ�idx0��һ����ֵ������û���ظ�Ԫ�ء������Ľ����������������
c = unique(label2);
P_size = length(p);% �˹���ǵĴصĸ���
C_size = length(c);% �㷨����Ĵصĸ���

    % Pid,Rid���������ݣ���i�з������ݴ�����������ڵ�i����
Pid = double(ones(P_size,1)*label1'== p*ones(1,N) );
Cid = double(ones(C_size,1)*label2'== c*ones(1,N) );
CP = Cid*Pid';%P��C�Ľ���,C*P

Px = sum(Cid')/N;
Py = sum(Pid')/N;
Pxy = CP/N;
MImatrix = Pxy .* log2(Pxy ./(Px' * Py)+eps);
MI = sum(MImatrix(:));%
% Entropies
Hx = -sum(Px .* log2(Px + eps),2);
Hy = -sum(Py .* log2(Py + eps),2);
%Normalized Mutual information
NMI= 2 * MI / (Hx+Hy);



Pj = sum(CP,1);% ��������P��C�������еĸ���
Ci = sum(CP,2);% ��������C��P�������еĸ���
precision = CP./( Ci*ones(1,P_size) );
recall = CP./( ones(C_size,1)*Pj );
F = 2*precision.*recall./(precision+recall);
    % �õ�һ���ܵ�Fֵ
FM=sum( (Pj./sum(Pj)).*max(F));
Accuracy = sum(max(CP,[],2))/N;
    
n=length(label1);
cp=crosstab(label1,label2);
cp(P_size+1,:)=sum(cp);
cp(:,P_size+1)=sum(cp,2);
%%����ARI
cp1=cp(1:P_size,1:P_size);
r0=sum(sum((cp1.*(cp1-1))./2));
cp2=cp(1:P_size,1+P_size)';
r1=sum((cp2.*(cp2-1))./2);
cp3=cp(1+P_size,1:P_size);
r2=sum((cp3.*(cp3-1))./2);
r3=(2*r1*r2)/(n*(n-1));
ARI=(r0-r3)/(0.5*(r1+r2)-r3);

end
    
    
   
  
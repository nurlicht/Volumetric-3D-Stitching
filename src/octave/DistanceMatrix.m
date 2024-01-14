function [S2, N]=DistanceMatrix(A, k)
    N_orient=size(A,2); %A: Images
    A=A-repmat(mean(A,2),1,size(A,2));
    A=A./repmat(sqrt(sum(A.^2,1)),size(A,1),1); %Unity-norm Images
    A=real(acos(A'*A)).^2;   %A <-- Round distance of Images ^2
    S2=randn(N_orient,k);   %Initializing kNN Distance ^2
    N=randn(N_orient,k);    %Initializing kNN Indices
    for cntr=1:N_orient
        [YY,II]=sort(A(cntr,:),'ascend');
        N(cntr,:)=II(1:k);
        S2(cntr,:)=YY(1:k);
    end
end


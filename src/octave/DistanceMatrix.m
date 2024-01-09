function []=DistanceMatrix(k)
    Wait_Bar=waitbar(0,'Loading snapshots');drawnow;
    A=importdata('../../artifacts/Images.mat');
    close (Wait_Bar);drawnow;
    N_orient=size(A,2); %A: Images
    A=A-repmat(mean(A,2),1,size(A,2));
    Wait_Bar=waitbar(0,'Calculating the {\bfRound} Distance Matrix');
    drawnow;
    A=A./repmat(sqrt(sum(A.^2,1)),size(A,1),1); %Unity-norm Images
    A=real(acos(A'*A)).^2;   %A <-- Round distance of Images ^2
    S2=randn(N_orient,k);   %Initializing kNN Distance ^2
    N=randn(N_orient,k);    %Initializing kNN Indices
    Wait_Bar=waitbar(0,Wait_Bar,['Sorting distances (kNN, k=' ...
        num2str(k) ')']);drawnow;
    for cntr=1:N_orient
        [YY,II]=sort(A(cntr,:),'ascend');
        N(cntr,:)=II(1:k);
        S2(cntr,:)=YY(1:k);
        if ~mod(cntr,50)
            waitbar(cntr/N_orient,Wait_Bar);drawnow;
        end
    end
    clear A;
    save '../../artifacts/S2.mat' S2 -v6;
    save '../../artifacts/N.mat' N -v6;
    close(Wait_Bar);drawnow;
end


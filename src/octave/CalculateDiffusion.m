function [Ps, Lambda, c0, c]=CalculateDiffusion(S2, N, Rotation_Axis, k,Epsilon,aPrioriFlag)
    pkg load statistics
    pkg load signal
    pkg load optim

    Rotation_R=Axis2RotUnfold(Rotation_Axis);
    [Ps,Lambda]=Diffusion_Coordinate(S2,N,Epsilon);
    if isnan(Lambda)
        disp(['Problem in convergence of eigenvalues with Epsilon=' ...
            num2str(Epsilon) ' and k=' num2str(k)]);
    elseif ~isreal(Ps(:))
        disp(['Complex eigenvector found with Epsilon=' ...
            num2str(Epsilon) ' and k=' num2str(k)]);
    else
        [c,c0]=DM_Fit_All(Ps,Rotation_R,aPrioriFlag);
    end
end
function [c,c0]=DM_Fit_All(Ps,Rotation_R,aPrioriFlag)
    if aPrioriFlag
        aPrioriText=' {\bfwith} ';
    else
        aPrioriText=' {\bfwithout} ';
    end
    Ps=Ps(:,2:end);
    Rotation_R=Rotation_R';
    if aPrioriFlag
        c0=Ps\Rotation_R;
        c=c0;
    else
        c0_1D=rand(81,1);
        c_scale=5;
        LB=[];
        UB=[];
        Options=Opt_param(c_scale,c0_1D);
        [c_1D,ResNorm,Residual,ExitFlag]= ...
            lsqnonlin(@(C)OR_Func(C,Ps'),c0_1D*c_scale,LB,UB,Options);
        c=pinv(reshape(c_1D,[9 9]));
        c0=pinv(reshape(c0_1D,[9 9]));
    end
    Recon_R=Ps*c;
    N=size(Ps,1);
    for cntr=1:N
        Temp_00=PolarDecompose(reshape(Recon_R(cntr,:),[3 3]));
        Recon_R(cntr,:)=Temp_00(:);
    end
    Q_1=rand(4,N);      %Initializing "Eestimated" Quaternions of rotations
    Q_2=rand(4,N);      %Initializing "Known" Quaternions of rotations
    for cntr=1:N
        r0=reshape(Recon_R(cntr,:),[3 3]);
        Q_1(1:4,cntr)=RotMat2Quat(r0);
        r0=reshape(Rotation_R(cntr,:),[3 3]);
        Q_2(1:4,cntr)=RotMat2Quat(r0);
    end
    TempA=real(acos(abs(Q_1'*Q_1)));    %Pairwise geodesic distances
    TempB=real(acos(abs(Q_2'*Q_2)));    %Pairwise geodesic distances
    Sigma_T=(180/pi)*2*sum(abs(TempA(:)-TempB(:)))/(N*(N-1));
    disp('Measure of error in Relative Orientations of All Pairs')
    disp(['Sigma_All_Pairs: ' num2str(Sigma_T) ' degrees'])
end


%% Calculating the diffusion coordinate
function W=S2_2_W_Matrix(S2,N,Epsilon)
    [N_orient,d]=size(S2);
    S2_Max=median(S2(:,5));

    Index=1;
    Dist_Max=min(S2(:,d));
    while max(S2(:,Index)) < Dist_Max
        Index=Index+1;
    end
    Dist_Thr=max(S2(:,Index-1));
    S2(S2 > Dist_Thr) = inf;

    Epsilon=Epsilon*S2_Max;
    S2=double(S2);
    N=double(N);
    W=sparse(repmat((1:N_orient)',1,d),N,exp(-S2/Epsilon));
    %W=(W+W')/2;
end
function [Ps,Lambda]=Diffusion_Coordinate(S2,N,Epsilon)
    [N_orient,~]=size(S2);
    [P_ep,D]=DM_Cov(AnIsoNorm(S2_2_W_Matrix(S2,N,Epsilon)));

    opts.disp=0;    %2
    opts.v0=1-5e-4*(0:(N_orient-1))';
    [Ps,Lambda,Eigen_Flag]=eigs(P_ep,10,'LM',opts);
    if ~Eigen_Flag
        [Lambda,Index]=sort(diag(Lambda),'descend');
        Ps=Ps(:,Index);
        Lambda=Lambda(Index);
    else
        disp('Error in eigenvalue calculation');
        Lambda=nan;
    end
    for cntr=1:size(Ps,2)
        Ps(:,cntr)=Ps(:,cntr)*Lambda(cntr);
    end
    Ps=D*Ps;
end
function W=AnIsoNorm(W)
    N_orient=size(W,1);
    Min=1/N_orient;
    Q_Alpha=sum(W,2);
    Q_Alpha(Q_Alpha < Min ) = Min;
    Q_Alpha=spdiags(1./Q_Alpha,0,N_orient,N_orient);
    W=Q_Alpha*W*Q_Alpha;
    W=(W+W')/2;
end
function [W,D]=DM_Cov(W)
    N_orient=size(W,1);
    Min=1/N_orient;
    D=sum(W,2);
    D(D < Min ) = Min;
    D=spdiags(1./sqrt(D),0,N_orient,N_orient);
    W=D*W*D;
    W=(W+W')/2;
end

%% Setting the nonlinear optimization parameters
function Options=Opt_param(varargin)
    c_scale=varargin{1};
    Options=optimset();
    Options.TolFun=1e-20*c_scale;
    Options.DiffMaxChange=1e5*c_scale;
    Options.Display='final';
    Options.MaxFunEvals=1e8;
    Options.MaxIter=250;   %1000
    Options.PlotFcns=@optimplotresnorm;
    %Options.PlotFcns=@optimplotx;
    Options.Algorithm='trust-region-reflective';
%    Options.Algorithm='levenberg-marquardt';
end

%% Imposing the rotation matrix constraints to find {c}
function G_Functional=OR_Func(c,PsMod)
    %Note: PsMod=Ps(:,1+(1:NPsi))'
    r=size(PsMod,2);
    r_MP5=1/sqrt(r);
    N_c=9;
    c_Temp=reshape(c,[N_c N_c]);

    Temp=zeros(N_c,N_c);    %Memory allocation
    Temp2=randn(N_c^2,1);   %Memory allocation
    I=eye(3);

    G_Functional=zeros(r,1);
    R_Big=c_Temp*PsMod;
    for cntr_l=1:r
        R=reshape(R_Big(:,cntr_l),[3 3]);
        Temp=R'*R-I;
        Temp2=Temp(:);
        G_Functional(cntr_l)=sqrt(Temp2'*Temp2)+abs(det(R)-1);%L2
        %G_Functional(cntr_l)=sum(abs(Temp(:)))+abs(det(R)-1);  %L1
    end
    G_Functional=sqrt(G_Functional)*r_MP5;
end


%% Major functions
function []=DM(varargin)
    %Initialization
    close all force;
    clear all;
    clc;

    %Similarity Matrix parameters
    k=150;
    Epsilon=0.7;

    %Whether to use known rotation matrices to estimate c(9,9)
    aPrioriFlag=0;

    %Main menu
    switch menu('Please select the operation',...
            'Generation of snapshots',...
            'Calculation of Distance Matrix',...
            'Estimation of Diffusion Coordinate');
        case 1
            Rotated_Images();
        case 2
            Calculate_Distance_Matrix(k);
            disp(['k=' num2str(k)]);
        case 3
            Calculate_Diffusion(k,Epsilon,aPrioriFlag);
            disp(['k=' num2str(k) ', Epsilon=' num2str(Epsilon)]);
    end
end
function Calculate_Distance_Matrix(k)
    Wait_Bar=waitbar(0,'Loading snapshots');drawnow;
    A=importdata('./Images.mat');
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
    save S2.mat S2 -v7.3;
    save N.mat N -v7.3;
    close(Wait_Bar);drawnow;
end
function []=Calculate_Diffusion(k,Epsilon,aPrioriFlag)
    Wait_Bar=waitbar(0,'Loading matrices');drawnow;
    S2=importdata('./S2.mat');
    N=importdata('./N.mat');
    Rotation_Axis=importdata('./Axis.mat');
    Rotation_R=Axis2RotUnfold(Rotation_Axis);
    Wait_Bar=waitbar(0.5,Wait_Bar,...
        'Finding the Diffusion Coordinate \psi');drawnow;
    [Ps,Lambda]=Diffusion_Coordinate(S2,N,Epsilon);
    close(Wait_Bar);drawnow;
    if isnan(Lambda)
        disp(['Problem in convergence of eigenvalues with Epsilon=' ...
            num2str(Epsilon) ' and k=' num2str(k)]);
    elseif ~isreal(Ps(:))
        disp(['Complex eigenvector found with Epsilon=' ...
            num2str(Epsilon) ' and k=' num2str(k)]);
    else
        Plot_Diffusion_Coordinate(Ps,Lambda,Rotation_Axis);
        save Ps.mat Ps -v7.3;
        save Lambda.mat Lambda -v7.3;
        [c,c0]=DM_Fit_All(Ps,Rotation_R,aPrioriFlag);
        save c0.mat c0 -v7.3;
        save c.mat c -v7.3;
    end
end
function [c,c0]=DM_Fit_All(Ps,Rotation_R,aPrioriFlag)
    if aPrioriFlag
        aPrioriText=' {\bfwith} ';
    else
        aPrioriText=' {\bfwithout} ';
    end
    WB=waitbar(0,['Finding C' aPrioriText 'known rotations']);drawnow;
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
    figure;imagesc(c);axis equal;axis off;title('Matrix C(9,9)');colorbar;
    WB=waitbar(0,WB,['Assessment of estimation (' aPrioriText ...
        'known rotations)']);drawnow;
    Recon_R=Ps*c;
    N=size(Ps,1);
    for cntr=1:N
        Temp_00=Polar_Decompose(reshape(Recon_R(cntr,:),[3 3]));
        Recon_R(cntr,:)=Temp_00(:);
    end
    Q_1=rand(4,N);      %Initializing "Eestimated" Quaternions of rotations
    Q_2=rand(4,N);      %Initializing "Known" Quaternions of rotations
    for cntr=1:N
        r0=reshape(Recon_R(cntr,:),[3 3]);
        Q_1(1:4,cntr)=RotMat2Quat(r0);
        r0=reshape(Rotation_R(cntr,:),[3 3]);
        Q_2(1:4,cntr)=RotMat2Quat(r0);
        if ~mod(cntr,5)
            WB=waitbar(cntr/N,WB,'Forming rotation arrays');drawnow;
        end
    end
    WB=waitbar(0,WB,'Assessment of estimated {\bfQuaternions}');drawnow;
    TempA=real(acos(abs(Q_1'*Q_1)));    %Pairwise geodesic distances
    TempB=real(acos(abs(Q_2'*Q_2)));    %Pairwise geodesic distances
    Sigma_T=(180/pi)*2*sum(abs(TempA(:)-TempB(:)))/(N*(N-1));
    disp('Measure of error in Relative Orientations of All Pairs')
    disp(['Sigma_All_Pairs: ' num2str(Sigma_T) ' degrees'])
    close(WB);drawnow;
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

%% Miscellaneous plots of the diffusion coordinate
function Plot_Diffusion_Coordinate(Ps,Lambda,Rotation)
    Plot_Eigenvalue_Eigenvector(Ps,Lambda)
    Plot_Individual_EV(Ps,Lambda)
    Plot_Diffusion_Statistics(Ps)
    Plot_Corr_Ps(Ps);
    Plot_Psi234_ColorRot(Ps,Lambda,Rotation,'Axis')
end
function Plot_Eigenvalue_Eigenvector(Ps,Lambda)
    figure
    set(gca,'NextPlot','replacechildren');
    subplot(211)
    imagesc(Ps)
    colorbar
    title('Diffusion map eigenfunctions {\psi_i}')
    subplot(212)
    plot(Lambda,'-*')
    colorbar
    title('Diffusion map eigenvalues')
end
function Plot_Diffusion_Statistics(Ps)
    figure
    set(gca,'NextPlot','replacechildren');
    subplot(221)
    bar(Ps(:));
    title(['First 10 \psi, min=' num2str(min(Ps(:))) ...
        ', Max=' num2str(max(Ps(:)))]);
    PsNorm=Ps(:,1).^2;
    for cntr=2:10
        PsNorm=PsNorm+Ps(:,cntr).^2;
    end
    PsNorm=sqrt(PsNorm);
    subplot(222)
    bar(PsNorm);
    title(['10-element norm, min=' num2str(min(PsNorm)) ...
        ', Max=' num2str(max(PsNorm))]);
    subplot(223)
    histfit(PsNorm,500,'logistic');
    title('10-element norm histogram');
    subplot(224)
    Index=(1:size(PsNorm,1))';
    Index=2*pi*Index/max(Index);
    polar(Index,PsNorm,'b');
    title('10-element norm polar histogram');
    hold on
    polar(Index,mean(PsNorm)*ones(size(PsNorm)),'r--');
    hold off
end
function Plot_Individual_EV(Ps,Lambda)
    figure
    set(gca,'NextPlot','replacechildren');
    for cntr=1:9
        subplot(3,3,cntr)
        plot(Ps(:,cntr+1).*Lambda(cntr+1))
        title(['\psi_' num2str(cntr+1)])
    end
end
function Plot_Psi234_ColorRot(Ps,Lambda,Rot,Text)
    for cntr=1:size(Rot,1)
        figure
        set(gca,'NextPlot','replacechildren');

        subplot(221)
        scatter(Ps(:,2).*Lambda(2),Ps(:,3).*Lambda(3),20,Rot(cntr,:))
        title(['Phase plane: \psi_2 vs. \psi_3 - Color-coded by ' ...
            Text ' ' num2str(cntr)]);
        colorbar

        subplot(222)
        scatter(Ps(:,2).*Lambda(2),Ps(:,4).*Lambda(4),20,Rot(cntr,:))
        title(['Phase plane: \psi_2 vs. \psi_4 - Color-coded by ' ...
            Text ' ' num2str(cntr)]);
        colorbar

        subplot(223)
        scatter(Ps(:,3).*Lambda(3),Ps(:,4).*Lambda(4),20,Rot(cntr,:))
        title(['Phase plane: \psi_3 vs. \psi_4 - Color-coded by ' ...
            Text ' ' num2str(cntr)]);
        colorbar

        subplot(224)
        scatter3(Ps(:,2).*Lambda(2),Ps(:,3).*Lambda(3),...
            Ps(:,4).*Lambda(4),20,Rot(cntr,:))
        title(['Phase plane: \psi_2-\psi_3-\psi_4 - Color-coded by ' ...
            Text ' ' num2str(cntr)]);
        colorbar
    end
    drawnow;
end
function Plot_Corr_Ps(Ps)
    figure
    subplot(4,1,1)
    plot(real(xcov(Ps(:,2),Ps(:,2))))
    title('Covariance of \psi_2 and \psi_2');
    legend('Auto correlation')
    ylim([-0.2 1.1])
    subplot(4,1,2)
    plot(real(xcov(Ps(:,2),Ps(:,3))))
    title('Covariance of \psi_2 and \psi_3');
    legend('Cross correlation')
    ylim([-0.2 1.1])
    subplot(4,1,3)
    plot(real(xcov(Ps(:,2),Ps(:,4))))
    title('Covariance of \psi_2 and \psi_4');
    legend('Cross correlation')
    ylim([-0.2 1.1])
    subplot(4,1,4)
    plot(real(xcov(Ps(:,3),Ps(:,4))))
    title('Covariance of \psi_3 and \psi_4');
    legend('Cross correlation')
    ylim([-0.2 1.1])
    drawnow;
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


%% Generating snapshots
function []= Rotated_Images()
    % Input parameters
    N_loop=28^3;   %Cube of an "even" integer
    Experiment=Experiment_Parameters();

    % Loading the object
    WaitBar=waitbar(0,'Generating the 3D object');drawnow;
    Protein=Load_Protein();
    
    % Separating Object values (at voxels) and space coordinates
    Grid_3D=Protein.Grid_3D;
    ED=Protein.ED;
    clear Protein;

    % Imaging w/ initial orientation
    N_p=Experiment.N_p;
    N_p2=N_p^2;

    % Loop
    WaitBar=waitbar(0,WaitBar,'Generating Rotation Matrices');drawnow;
    R=AllRotMatrices(N_loop);
    
    R_size=[3 3];
    WaitBar=waitbar(0,WaitBar,'Memory allocation');drawnow;
    Images=randn(N_p2,N_loop);  %Memory allocation
    WaitBar=waitbar(0,WaitBar,['Generating ' num2str(N_loop) ...
        ' snapshots']);drawnow;

    [Lambda,zD,Width,N]=Extract_ExpParam(Experiment);
    [Length,Number]=Extract_Coordinates(Grid_3D);
    
    [~,k]=FourierScaledAxes(Number,Length);
  
    % Camera coordinate (k-space)
    [Camera_x,Camera_y]=meshgrid((Width/(N-1))*((1:N)-(N+1)/2));
    Circle_Index=((Camera_x.^2+Camera_y.^2) > (Width/2)^2);    
    Temp=Lambda*sqrt(Camera_x.^2+Camera_y.^2+zD^2);
    Q_x=Camera_x./Temp;
    Q_y=Camera_y./Temp;
    Q_z=(zD./Temp-1/Lambda);

    for cntr=1:N_loop
        ED_rot=RotateStructureIndex(ED,reshape(R(:,cntr),R_size));
        Camera_I=interp3(k.x,k.y,k.z,Shift_FFT(abs(fftn(ED_rot))),...
            Q_x,Q_y,Q_z,'linear',0);
        Camera_I(Circle_Index)=0;
        Images(:,cntr)=reshape(Camera_I,[N_p2 1]);
        if ~mod(cntr,50)
            waitbar(cntr/N_loop,WaitBar,['Generating snapshot ' ...
                num2str(cntr) ' out of ' num2str(N_loop)]);drawnow;
        end
    end
    WaitBar=waitbar(0,WaitBar,'Saving snapshots');drawnow;
    save Images.mat Images  '-v7.3';
    close(WaitBar);drawnow;
end
function Experiment=Experiment_Parameters()
    N_P_NoBin=1024;
    Experiment=struct;
    Experiment.N_p=63; %number of pixels along each coordinate
    Experiment.Pixel=75e-6;
    Experiment.zD=0.5;    %0.738
    Experiment.Lambda=2*1e-9;    %Doubled! 2*1.03e-9
    Experiment.SuperPixel=Experiment.Pixel*(N_P_NoBin/Experiment.N_p);
    Experiment.Width=Experiment.SuperPixel*Experiment.N_p;
end
function [Lambda,zD,Width,N]=Extract_ExpParam(Experiment)
    Lambda=Experiment.Lambda;
    zD=Experiment.zD;
    Width=Experiment.Width;
    N=Experiment.N_p;
end


%% Synthesing a 3D object in real-space
function Protein=Load_Protein(varargin)
    Protein_Source=2;
    switch Protein_Source
        case 1
            Protein_File='Protein.mat';
            WaitBar=waitbar(0,'Loading protein data');
            pause(1e-3);
            Protein=load(Protein_File);
            Protein=Protein.Protein;
            delete(WaitBar);
            pause(1e-3);
        case 2
            if nargin==1
                Protein=IR_3D(varargin{1});
            elseif ~nargin
                Protein=IR_3D();
            end
    end
end
function Protein=IR_3D(varargin)
    if ~nargin
        close all;
        %clear all;
        clc;
        pause(1e-6);
    end
    N1=31;
    N2=N1;
    N3=N1;
    U=((1:N1)-(N1+1)/2)/(N1/2);
    V=((1:N2)-(N2+1)/2)/(N2/2);
    W=((1:N3)-(N3+1)/2)/(N3/2);
    [x,y,z]=meshgrid(U,V,W);
    A=x/0.47;
    B=y/0.37;
    C=z/0.29;
    if nargin==1
        Scale=varargin{1};
        A=A/Scale;
        B=B/Scale;
        C=C/Scale;
    end
    F=(1-0.4*((A-0.15).^2+(B+0.2).^2+(C-0.1).*(A-0.15).*(B+0.2)));
    F(  (cos(20*pi*(x-z-0.2).*abs(y+z+0.1).*abs(z-0.3)) < 0.2) | ...
        (A.^2+B.^2+C.^2 >1)  | (F<0) )=0;
    if ~nargin
        Protein=struct;
        Protein.ED=F;
        clear F;
        Factor=2e-7; %3e-7
        Protein.Grid_3D.x=x*Factor;
        Protein.Grid_3D.y=y*Factor;
        Protein.Grid_3D.z=z*Factor;
    else
        Protein=F;
    end
end
function F=Shift_FFT(F)
    N=size(F);
    Nh=(N-1)/2;
    for cntr=1:3
        Index{cntr}=[(Nh(cntr)+1):N(cntr),1:Nh(cntr)];
    end
    F=F(Index{1},Index{2},Index{3});
end
function [Nyquist,k]=FourierScaledAxes(Number,Length)
    Nyquist=struct;
    k=struct;
    [Nyquist.x,k.x]=FourierScaledAxis(Number.x,Length.x);
    [Nyquist.y,k.y]=FourierScaledAxis(Number.y,Length.y);
    [Nyquist.z,k.z]=FourierScaledAxis(Number.z,Length.z);
    [k.x,k.y,k.z]=meshgrid(k.x,k.y,k.z);
end
function [Nyquist,k]=FourierScaledAxis(Number,Length)
    d=Length/(Number-1);
    Nyquist=0.5/d;
    N1=(Number-1)/2;
    k=(-N1:N1)*(2*Nyquist/Number);
end
function [Length,Number]=Extract_Coordinates(Grid_3D)
    Length=struct;
    Temp=Grid_3D.x(:);
    Length.x=max(Temp)-min(Temp);
    Temp=Grid_3D.y(:);
    Length.y=max(Temp)-min(Temp);
    Temp=Grid_3D.z(:);
    Length.z=max(Temp)-min(Temp);
    
    Number=struct;
    [Number.x,Number.y,Number.z]=size(Grid_3D.x);
end


%% Rotations
function Protein=RotateStructureIndex(F,R)
    N=max(size(F));
    [x,y,z]=meshgrid(((1:N)-(N+1)/2)/(N/2));
    Q=R*[x(:),y(:),z(:)]';
    Qx=reshape(Q(1,:),[N N N]);
    Qy=reshape(Q(2,:),[N N N]);
    Qz=reshape(Q(3,:),[N N N]);
    Protein=interp3(x,y,z,F,Qx,Qy,Qz,'linear',0);
end
function Q=Uniform_SO3_Hopf(N_orient)
    n1=round(N_orient^(1/3));
    N_orient=n1^3;
    N=[n1 n1 n1];
   
    Psi_=(2*pi)*linspace(0,1,N(1)+1);
    %Theta_=acos(Factor*linspace(1,-1,N(2)+1));
    Theta_=acos(linspace(1,-1,N(2)));
    Phi_=(2*pi)*linspace(0,1,N(3)+1);
        
    Psi_=Psi_(1:N(1));
    Theta_=Theta_(1:N(2));
    Phi_=Phi_(1:N(3));
    
    Psi_=Psi_-mean(Psi_);
    Theta_=Theta_-mean(Theta_)+(pi/2);
    Phi_=Phi_-mean(Phi_);
    
    [Psi,Theta,Phi]=meshgrid(Psi_,Theta_,Phi_);
    Psi=Psi(:);
    Theta=Theta(:);
    Phi=Phi(:);
    
    Index=1:N_orient;
    Psi=Psi(Index);
    Theta=Theta(Index);
    Phi=Phi(Index);
        
    Q=[cos(Theta/2).*cos(Psi/2),...
      sin(Theta/2).*sin(Phi+Psi/2),...
      sin(Theta/2).*cos(Phi+Psi/2),...
      cos(Theta/2).*sin(Psi/2)]';
    Q=UnitMagPos(Q);
end
function Q=UnitMagPos(Q)
    Norm=zeros(1,size(Q,2));
    for cntr=1:4
        Norm=Norm+Q(cntr,:).^2;
    end
    Norm=sqrt(Norm);
    Sign=sign(Q(1,:));
    for cntr=1:4
        Q(cntr,:)=Q(cntr,:).*Sign./Norm;
    end
end
function R=Axis2RotMatBatch(Axis)
    N=size(Axis,2);
    R=zeros(9,N);
    for cntr=1:N
        R(:,cntr)=reshape(Axis2RotMat(Axis(:,cntr)),[9 1]);
    end
    if sum(isnan(R(:)))
        disp('Nan in rotation matrix batch')
    end
end
function Q=Uniform_SO3_PDF(N)
    Q=randn(4,N);
    Norm=sqrt(Q(1,:).^2+Q(2,:).^2+Q(3,:).^2+Q(4,:).^2);
    for cntr=1:4
        Q(cntr,:)=Q(cntr,:)./Norm;
    end
end
function R=AllRotMatrices(N)
    Mode=2;
    switch Mode
        case 1  %Random
            Q=Uniform_SO3_PDF(N);
            Axis=Quat2Axis(Q);
        case 2  %Hopf
            Q=Uniform_SO3_Hopf(N);           
            Axis=Quat2Axis(Q);
    end
    R=Axis2RotMatBatch(Axis);
    save Axis.mat Axis  '-v7.3';
end
function R_orth=Polar_Decompose(R)
    [U,~,V]=svd(R);
    R_orth=U*V';
end
function R=Axis2RotUnfold(Axis)
    N_R2=9;
    N_orient=size(Axis,2);
    R=zeros(N_R2,N_orient);
    for cntr=1:N_orient
        Temp=Axis2RotMat(Axis(1:3,cntr));
        R(1:N_R2,cntr)=Temp(:);
    end
end
function Q=RotMat2Quat(R)
    Axis=RotMat2Axis(R);
    Q=Axis2Quat(Axis);
end
function R=Axis2RotMat(Axis)
    Theta=norm(Axis);
    Axis=Axis/Theta;
    a=cos(Theta);
    la=1-cos(Theta);
    b=sin(Theta);
    m=Axis(1);
    n=Axis(2);
    p=Axis(3);
    R=[a+m^2*la, m*n*la-p*b, m*p*la+n*b; ...
        n*m*la+p*b, a+n^2*la, n*p*la-m*b; ...
        p*m*la-n*b, p*n*la+m*b, a+p^2*la];
end
function Q=Axis2Quat(Axis)
    if size(Axis,1) > 3
        Axis=Axis';
    end
    N=size(Axis,2);
    Q=zeros(4,N);
    for cntr=1:N
        Temp=Axis(:,cntr);
        Theta=norm(Temp);
        if Theta
            Q(:,cntr)=[cos(Theta/2);sin(Theta/2)*(Temp/Theta)];
        else
            Q(:,cntr)=[1;0;0;0];
        end
    end
end
function Axis=RotMat2Axis(R)
    x=R(3,2)-R(2,3);
    y=R(1,3)-R(3,1);
    z=R(2,1)-R(1,2);
    r_2sin=norm([x,y,z]);
    if r_2sin
        Theta=atan2(r_2sin,trace(R)-1);
        Axis=(Theta/r_2sin)*[x;y;z];
    elseif R==eye(3)
        Axis=[0;0;0];
    else
        disp('Problem with the rotation matrix')
        R
    end
end
function Axis=Quat2Axis(Q)
    N=size(Q,2);
    Axis=zeros(3,N);
    for cntr=1:N
        Axis(:,cntr)=Quat2AxisSingle(Q(:,cntr));
    end
end
function Axis=Quat2AxisSingle(Q)
    Angle=real(2*acos(abs(Q(1))));
    if ~Angle
        Axis=zeros(3,1);
    else
        Axis_norm=Q(2:4)/norm(Q(2:4));
        Axis=Angle*Axis_norm;
    end
end

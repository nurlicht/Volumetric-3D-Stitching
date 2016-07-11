%% Documentation

%Seemingly good rewriting of the code for image generation, S2 calculation,
%and a prori estimation. Details to be verified.
%Seemingly wrong and to-be-vcompletely-verified writing of normal code


%%
function []=DM_1(varargin)
    %Initialization
    close all force;
    clear all;
    clc;

    %Similarity Matrix parameters
    k=150;
    Epsilon=0.7;

    %Whether to use known rotation matrices to estimate c(9,9)
    aPrioriFlag=1;  

    %Main menu
    switch menu('',...
            'Generate snapshots',...
            'Calculate the Distance Matrix',...
            'Calculate the Diffusion Coordinate + C(9,9)'...
            );
        case 1
            Rotated_Images();
        case 2
            Calculate_Distance_Matrix(k);
        case 3
            Calculate_Diffusion(k,Epsilon,aPrioriFlag);
    end
end
function Calculate_Distance_Matrix(k)
    Wait_Bar=waitbar(0,'Loading snapshots');drawnow;
    A=double(importdata('./Images.mat'));
    close (Wait_Bar);drawnow;
    N_orient=size(A,2); %A: Images
    A=A-repmat(mean(A,2),1,size(A,2));
    Wait_Bar=waitbar(0,'Calculating the {\bfRound} Distance Matrix');drawnow;
    A=A./repmat(sqrt(sum(A.^2,1)),size(A,1),1); %Unity-norm Images
    A=real(acos(A'*A)).^2;   %A <-- Round distance of Images ^2
    S2=randn(N_orient,k);   %Initializing kNN Distance ^2
    N=randn(N_orient,k);    %Initializing kNN Indices
    Wait_Bar=waitbar(0,Wait_Bar,['Sorting distances (kNN, k=' num2str(k) ')']);drawnow;
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
    Wait_Bar=waitbar(0.5,Wait_Bar,'Finding the Diffusion Coordinate \psi');drawnow;
    [Ps,Lambda]=Diffusion_Coordinate(S2,N,Epsilon);
    close(Wait_Bar);drawnow;
    if isnan(Lambda)
        disp(['Problem in convergence of eigenvalues with Epsilon=' ...
            num2str(Epsilon) ' and k=' num2str(k)]);
    elseif ~isreal(Ps(:))
        disp(['Complex eigenvector found with Epsilon=' ...
            num2str(Epsilon) ' and k=' num2str(k)]);
    else
        %Plot_Diffusion_Coordinate(Ps,Lambda,Rotation_Axis);
        save Ps.mat Ps -v7.3;
        save Lambda.mat Lambda -v7.3;
        if aPrioriFlag
            WB=waitbar(0,'Finding the C matrix {\bfwith} a priori orientations');drawnow;
            [Sigma,Median,c]=DM_Fit(Ps,Axis2RotUnfold(Rotation_Axis));
        else
            WB=waitbar(0,'Finding the C matrix {\bfwithout} a priori orientations');drawnow;
            c=zeros(9,9);   %Possibly an educated guess
            [Recon,Rotation]=Recon_Matrices(Ps,c,Param,Rotation);
            save Recon.mat Recon -v7.3 ;
        end
        save c.mat c -v7.3;
        save Sigma.mat Sigma -v7.3;
        save Median.mat Median -v7.3;
        close(WB);drawnow;
    end
end
function [Sigma,Median,c]=DM_Fit(Ps,Rotation_R)
    A=Ps(:,2:end);
    B=Rotation_R';
    c=A\B;
    Temp=A*c;
    N=size(Ps,1);
    for cntr=1:N
        Temp_00=Polar_Decompose(reshape(Temp(cntr,:),[3 3]));
        Temp(cntr,:)=Temp_00(:);
    end
    Angle_1=zeros(1,N);
    Q_1=zeros(4,N);
    Angle_2=zeros(1,N);
    Q_2=zeros(4,N);
    for cntr=1:N
        r0=reshape(Temp(cntr,:),[3 3]);
        Angle_1(cntr)=acos(0.5*(trace(r0)-1));
        Q_1(1:4,cntr)=RotMat2Quat(r0);
        r0=reshape(B(cntr,:),[3 3]);
        Angle_2(cntr)=acos(0.5*(trace(r0)-1));
        Q_2(1:4,cntr)=RotMat2Quat(r0);
    end
    Error2=(180/pi)*abs(abs(Angle_1-Angle_2));
    Sigma=std(Error2);
    Median=median(Error2);
    disp('Measures of error in eigenfunctions ONLY (using known orientations):')
    disp(['Sigma: ' num2str(Sigma) ' degrees, Median: ' num2str(Median) ' degrees'])
    TempA=real(acos(abs(Q_1'*Q_1)));
    TempB=real(acos(abs(Q_2'*Q_2)));
    Sigma_T=(180/pi)*2*sum(abs(TempA(:)-TempB(:)))/(N*(N-1));
    disp('Measure of error in Relative Orientations of All Pairs')
    disp(['Sigma_All_Pairs: ' num2str(Sigma_T) ' degrees'])
end

%% Recovery
function Recon_Metrics=Disp_Recon_Q(Epsilon,d,ResNorm,Ps_corr,Ps_Error,Error_Index)
    disp(['Epsilon=' num2str(Epsilon) ', d=' num2str(d) ', ResNorm=' num2str(ResNorm)]);
    [~,Index]=sort(Ps_corr,'descend');
    disp(['Reconstruction of eigenvectors: ' num2str(Ps_corr(Index(end))) ...
        ' < Correlation coefficient < ' num2str(Ps_corr(Index(1)))]);
    disp(['Total EV reconstruction error = ' num2str(100*Ps_Error) '%, '...
        'RMS angular distance error = ' num2str(Error_Index)]);

    Recon_Metrics=struct;
    Recon_Metrics.Epsilon=Epsilon;
    Recon_Metrics.d=d;
    Recon_Metrics.ResNorm=ResNorm;
    Recon_Metrics.CorrCoeffMin=Ps_corr(Index(end));
    Recon_Metrics.CorrCoeffMax=Ps_corr(Index(1));
    Recon_Metrics.EVRecPercError=100*Ps_Error;
    Recon_Metrics.RMSAngPercError=Error_Index;
    Recon_Metrics=struct2cell(Recon_Metrics);
end

%% Initialization
function [Flags,Param]=Initialize_Miscellaneous()
    Flags=struct;
    Flags.WB=1;
    Flags.Scale=1;
    Flags.Video=0;
    Flags.Status=1;
    
    Param=struct;
    Param.Path='./';
    Param.TransposeR=1;
    Param.Dim.N=3;
    switch Param.Dim.N
        case 1
            Param.Dim.Nc=2;
            Param.Dim.NPsi=2;
        case 3
            Param.Dim.Nc=9;
            Param.Dim.NPsi=9;
    end
    Param.Dim.NR=3;
    Param.Dim.Nc2=Param.Dim.Nc^2;
    Param.Diff.Alpha=1; % Alpha=1 for anisotropic diffusion map (Alpha=0: Random-walk)
    Param.Diff.c0_1D=2*randn(Param.Dim.Nc2,1)-1;
    Param.Diff.LB=[];
    Param.Diff.UB=[];
end

%% Calculating the W matrix
function W=S2_2_W_Matrix(S2,N,Epsilon)
    if length(Epsilon) > 1
        W=nan;
        return;
    end
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

%% Calculating the diffusion coordinate
function [Ps,Lambda]=Diffusion_Coordinate(S2,N,Epsilon,Alpha)
    Alpha=1;
    WB_Flag=1;
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

%% Finding the best fit of the C matrix
function [c1,ResNorm1,Residual1,ExitFlag1]=C_Fit(Ps,Param,Rotation)
    c0_1D=Param.Diff.c0_1D;
    c_scale=5;
    LB=Param.Diff.LB;
    UB=Param.Diff.UB;
    Param.Diff.c_scale=c_scale;
    PsMod=Ps(:,1+(1:Param.Dim.NPsi))';
    clear Ps;
    Options=Opt_param(c_scale,c0_1D);
        [c1,ResNorm1,Residual1,ExitFlag1]= ...
            lsqnonlin(@(c)OR_Func(c,PsMod,Rotation),c0_1D*c_scale,LB,UB,Options);
end



%% Miscellaneous plots of the diffusion coordinate
function Plot_All_Diff_Recov(W,P_ep,Ps,Lambda,Recon,Rotation,Param)
    figure
    plot(Recon.Residual)
    title(['Residual; Max=' num2str(Recon.ResNorm)]);
    Recon.Exit_Status=Decode_Status(Recon.ExitFlag);
    save Recon.mat Recon;
    %Plot_Diff_Matrices(W,P_ep);
    %Plot_Diffusion_Coordinate(Ps,Lambda,Rotation);
    Plot_orientation_recovery(Ps,Lambda,Recon,Rotation,Param);
end

function Plot_Diff_Matrices(W,P_ep)
    return
    
    figure
    set(gca,'NextPlot','replacechildren');

    subplot(221)
    spy(W)
    title('Gaussian kernel, W')
    colorbar
    daspect([1 1 1])

    subplot(222)
    spy(P_ep)
    title('Sparse transition probabbility, P_\epsilon')
    colorbar
    daspect([1 1 1])

    subplot(223)
    imagesc(W)
    title('Gaussian kernel, W')
    colorbar
    daspect([1 1 1])

    subplot(224)
    imagesc(P_ep)
    title('Sparse transition probabbility, P_\epsilon')
    colorbar
    daspect([1 1 1])
end

function Plot_Diffusion_Coordinate(Ps,Lambda,Rotation,Param)
    Angles=Rotation.Angles;
    Axis=Rotation.Axis;
    Axis_norm=Rotation.Axis_norm;
    Quaternion=Rotation.Q;
    Euler=Rotation.Euler;
    Wigner_D=Rotation.Wigner;
    
    Text=['d=' num2str(Param.d) ', \epsilon=' num2str(Param.Diff.Epsilon)];
%        ', Cond(W)=' num2str(Param.Diff.CondW) ', Cond(P_ep)=' ...
%        num2str(Param.Diff.CondP_ep)];
    
    Plot_Eigenvalue_Eigenvector(Ps,Lambda)
    title(Text);
    Plot_Individual_EV(Ps,Lambda)
    Plot_Diffusion_Statistics(Ps)
    Plot_Corr_Ps(Ps);

    Plot_Rot_Statistics(Axis,Axis_norm,Angles,Quaternion);

    Plot_Psi_ColorRot(Ps,Lambda,Axis,Axis_norm,Angles,Euler,Wigner_D,Quaternion);
    return;
    
    title(Text);
    Plot_Rot_ColorPsi(Ps,Lambda,Axis,Axis_norm,Euler,Quaternion);
end

function Plot_Rot_ColorPsi(Ps,Lambda,Axis,Axis_norm,Euler,Quaternion)
    Plot_2DRot_ColorPsi(Ps,Lambda,Quaternion(2:4,:),'Quaternion(2-4)')
    Plot_2DRot_ColorPsi(Ps,Lambda,Euler,'Euler')
    Plot_2DRot_ColorPsi(Ps,Lambda,Axis,'Axis')
    Plot_2DRot_ColorPsi(Ps,Lambda,Axis_norm,'Axis_norm')

    Euler_est=(WignerD2Euler(Ps(:,2:10)'.*repmat(Lambda(2:end),1,size(Ps,1))));
    Flip=4;
    switch Flip
        case 1
        case 2
            Euler_est(1,:)=-Euler_est(1,:);
            Euler_est(3,:)=-Euler_est(3,:);
        case 3
            Euler_est(1,:)=-Euler_est(1,:);
        case 4
            Euler_est(3,:)=-Euler_est(3,:);
    end
    
    figure
    for cntr=1:3
        subplot(1,3,cntr)
        scatter3(Euler(1,:),Euler(2,:),Euler(3,:),20,Euler_est(cntr,:))
        xlabel('Euler(1)')
        ylabel('Euler(2)')
        zlabel('Euler(3)')
        title(['Color-coded by Euler_e_s_t ' num2str(cntr)]) 
    end
    drawnow;
    
    Plot_3DRot_ColorPsi(Ps,Lambda,Quaternion(2:4,:),'Quaternion(2-4)')
    Plot_3DRot_ColorPsi(Ps,Lambda,Euler,'Euler')
    Plot_3DRot_ColorPsi(Ps,Lambda,Euler_est,'Euler_est')
    Plot_3DRot_ColorPsi(Ps,Lambda,Axis,'Axis')
    Plot_3DRot_ColorPsi(Ps,Lambda,Axis_norm,'Axis_norm')
end

function Plot_Psi_ColorRot(Ps,Lambda,Axis,Axis_norm,Angles,Euler,Wigner_D,Quaternion)
    %Plot_Psi234_ColorRot(Ps,Lambda,Quaternion,'Quaternion')
    Plot_Psi234_ColorRot(Ps,Lambda,Euler,'Euler')
    %Plot_Psi234_ColorRot(Ps,Lambda,Angles,'Angles')
    %Plot_Psi234_ColorRot(Ps,Lambda,Axis,'Axis')
    %Plot_Psi234_ColorRot(Ps,Lambda,Axis_norm,'Axis_norm')

    %Plot_3DPsi_ColorRot(Ps,Lambda,Quaternion,'Quaternion')
    %Plot_3DPsi_ColorRot(Ps,Lambda,Euler,'Euler')
    %Plot_3DPsi_ColorRot(Ps,Lambda,Angles,'Angles')
    %Plot_3DPsi_ColorRot(Ps,Lambda,Axis,'Axis')
    %Plot_3DPsi_ColorRot(Ps,Lambda,Axis_norm,'Axis_norm')
    
    
    return;

    

    

    
    Animate_Diffusion_Coordinate_3(Ps,Lambda,Angles,'angles');
    for cntr=1:3
        Animate_Diffusion_Coordinate_3(Ps,Lambda,Axis(cntr,:),['axis ' ...
            num2str(cntr)]);
    end
    for cntr=1:4
        Animate_Diffusion_Coordinate_3(Ps,Lambda,Quaternion(cntr,:),['Quaternion ' ...
            num2str(cntr)]);
    end
    
    
    pause(1e-3)
    return
    
    Plot_2DPsi_ColorRot(Ps,Lambda,Quaternion,'Quaternion')
    Plot_2DPsi_ColorRot(Ps,Lambda,Euler,'Euler')
    Plot_2DPsi_ColorRot(Ps,Lambda,Angles,'Angles')
    Plot_2DPsi_ColorRot(Ps,Lambda,Axis,'Axis')
    Plot_2DPsi_ColorRot(Ps,Lambda,Axis_norm,'Axis_norm')
    Plot_2DPsi_ColorRot(Ps,Lambda,Wigner_D,'Wigner_D')

    Plot_3DPsi_ColorRot(Ps,Lambda,Quaternion,'Quaternion')
    Plot_3DPsi_ColorRot(Ps,Lambda,Euler,'Euler')
    Plot_3DPsi_ColorRot(Ps,Lambda,Angles,'Angles')
    Plot_3DPsi_ColorRot(Ps,Lambda,Axis,'Axis')
    Plot_3DPsi_ColorRot(Ps,Lambda,Axis_norm,'Axis_norm')
    
    
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
    title(['First 10 \psi, min=' num2str(min(Ps(:))) ', Max=' num2str(max(Ps(:)))]);
    PsNorm=Ps(:,1).^2;
    for cntr=2:10
        PsNorm=PsNorm+Ps(:,cntr).^2;
    end
    PsNorm=sqrt(PsNorm);
    subplot(222)
    bar(PsNorm);
    title(['10-element norm, min=' num2str(min(PsNorm)) ', Max=' num2str(max(PsNorm))]);
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

function Plot_2DRot_ColorPsi(Ps,Lambda,Rot,Text)
    for Rot_cntr=1:size(Rot,1)
        figure
        set(gca,'NextPlot','replacechildren');
        switch Rot_cntr
            case 1
                M=1;
                N=2;
            case 2
                M=1;
                N=3;
            case 3
                M=2;
                N=3;
        end
        for cntr=2:10
            subplot(3,3,cntr-1)
            scatter(Rot(M,:),Rot(N,:),20,Ps(:,cntr).*Lambda(cntr))
            title([Text '(' num2str(M) ',' num2str(N) '), \psi_' ...
                num2str(cntr+1)])
        end
    end
end

function Plot_3DRot_ColorPsi(Ps,Lambda,Rot,Text)
    figure
    set(gca,'NextPlot','replacechildren');
    for cntr=2:10
        subplot(3,3,cntr-1)
        scatter3(Rot(1,:),Rot(2,:),Rot(3,:),20,Ps(:,cntr).*Lambda(cntr))
        title([Text ' coded by \psi_' num2str(cntr)])
    end
end

function Plot_Psi234_ColorRot(Ps,Lambda,Rot,Text)
    for cntr=1:size(Rot,1)
        figure
        set(gca,'NextPlot','replacechildren');

        subplot(221)
        scatter(Ps(:,2).*Lambda(2),Ps(:,3).*Lambda(3),20,Rot(cntr,:))
        title(['Phase plane: \psi_2 vs. \psi_3 - Color-coded by ' Text ' ' num2str(cntr)]);
        colorbar

        subplot(222)
        scatter(Ps(:,2).*Lambda(2),Ps(:,4).*Lambda(4),20,Rot(cntr,:))
        title(['Phase plane: \psi_2 vs. \psi_4 - Color-coded by ' Text ' ' num2str(cntr)]);
        colorbar

        subplot(223)
        scatter(Ps(:,3).*Lambda(3),Ps(:,4).*Lambda(4),20,Rot(cntr,:))
        title(['Phase plane: \psi_3 vs. \psi_4 - Color-coded by ' Text ' ' num2str(cntr)]);
        colorbar

        subplot(224)
        scatter3(Ps(:,2).*Lambda(2),Ps(:,3).*Lambda(3),Ps(:,4).*Lambda(4),20,Rot(cntr,:))
        title(['Phase plane: \psi_2-\psi_3-\psi_4 - Color-coded by ' Text ' ' num2str(cntr)]);
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

function Plot_Rot_Statistics(Axis,Axis_norm,Angles,Quaternion)
    N_Hist=20;
    figure
    subplot(5,2,1)
    hist(Axis(1,:),N_Hist)
    legend('Axis(1)')
    subplot(5,2,3)
    hist(Axis(2,:),N_Hist)
    legend('Axis(2)')
    subplot(5,2,5)
    hist(Axis(3,:),N_Hist)
    legend('Axis(3)')
    subplot(5,2,7)
    hist(Angles,N_Hist)
    legend('\Theta')
    subplot(5,2,9)
    hist(cos(Angles/2),N_Hist)
    legend('Cos(\Theta/2)')

    subplot(5,2,2)
    hist(Quaternion(2,:),N_Hist)
    legend('Quaternion(2)')
    subplot(5,2,4)
    hist(Quaternion(3,:),N_Hist)
    legend('Quaternion(3)')
    subplot(5,2,6)
    hist(Quaternion(4,:),N_Hist)
    legend('Quaternion(4)')
    subplot(5,2,8)
    hist(Quaternion(1,:),N_Hist)
    legend('Quaternion(1)')

    Sections_3D(Axis(1,:),Axis(2,:),Axis(3,:),Angles,8,'\Theta vs. Axes');
    Sections_3D(Axis(1,:),Axis(2,:),Axis(3,:),cos(Angles/2),8,...
        'Cos(\Theta/2) vs. Axes');
    Sections_3D(Axis_norm(1,:),Axis_norm(2,:),Axis_norm(3,:),Angles,8,...
        '\Theta vs. Normalized Axes');
    Sections_3D(Axis_norm(1,:),Axis_norm(2,:),Axis_norm(3,:),cos(Angles/2),8,...
        'Cos(\Theta/2) vs. Normalized Axes');
    Sections_3D(Quaternion(2,:),Quaternion(3,:),Quaternion(4,:),...
        Quaternion(1,:),8,'Quaternion(1) vs. Quaternions(2:4)');
    
    drawnow;
end

function Sections_3D(X,Y,Z,F,n,Text)
    X=X(:);
    Y=Y(:);
    Z=Z(:);
    F=F(:);
    Range=max(F)-min(F);
    Step=Range/n;
    [~,Hist_Value]=hist(F,n);
    n1=floor(sqrt(n+1));
    n2=ceil((n+1)/n1);
    Temp=max(n1,n2);
    n1=min(n1,n2);
    n2=Temp;
    figure;
    for cntr1=1:n1
        for cntr2=1:n2
            Cntr=(cntr1-1)*n2+cntr2;
            if Cntr <= n
                Index=find( abs(F-Hist_Value(Cntr)) <= (Step/2) );
                subplot(n1,n2,Cntr);
                scatter3(X(Index),Y(Index),Z(Index),20,F(Index));
                xlabel('x');ylabel('y');zlabel('z');
                title(['Values around ' num2str(Hist_Value(Cntr))]);
            end
        end
    end
    subplot(n1,n2,n+1)
    title(Text)
    drawnow;
end

function Plot_2DPsi_ColorRot(Ps,L,Rot,Color)
    Dim.x=2;
    Dim.y=2;
    for Rot_cntr=1:1%size(Rot,1)
        figure
        Cntr=1;
        for cntr1=2:3
            for cntr2=(cntr1+1):4
                    M=[cntr1,cntr2];
                    subplot(Dim.x,Dim.y,Cntr);
                    scatter(Ps(:,M(1)).*L(M(1)),Ps(:,M(2)).*L(M(2)), ...
                        20,Rot(Rot_cntr,:));
                    if cntr2==10
                        M2='1_0';
                    else
                        M2=num2str(M(2));
                    end
                    title(['\psi_' num2str(M(1)) '-\psi_' M2 ...
                        ', color: ' Color ' (' num2str(Rot_cntr) ')']);
                    colorbar
                    axis equal;
                    Cntr=Cntr+1;
            end
        end
        pause(1e-3);
    end
end

function Plot_3DPsi_ColorRot(Ps,L,Rot,Color)
    Dim.x=7;
    Dim.y=12;
    for Rot_cntr=1:size(Rot,1)
        figure
        Cntr=1;
        for cntr1=2:8
            for cntr2=(cntr1+1):9
                for cntr3=(cntr2+1):10
                    M=[cntr1,cntr2,cntr3];
                    subplot(Dim.x,Dim.y,Cntr);
                    scatter3(Ps(:,M(1)).*L(M(1)),Ps(:,M(2)).*L(M(2)), ...
                        Ps(:,M(3)).*L(M(3)),20,Rot(Rot_cntr,:));
                    if cntr3==10
                        M3='1_0';
                    else
                        M3=num2str(M(3));
                    end
                    title(['\psi_' num2str(M(1)) '-\psi_' num2str(M(2)) ...
                        '-\psi_' M3 ', color: ' Color ' (' ...
                        num2str(Rot_cntr) ')']);
                    colorbar
                    Cntr=Cntr+1;
                end
            end
        end
        pause(1e-3);
    end
end

function Animate_Diffusion_Coordinate_3(Ps,L,A,Color)
    %return;
    figure
    set(gca,'NextPlot','replacechildren');
    for cntr1=2:8
        for cntr2=(cntr1+1):9
            for cntr3=(cntr2+1):10
                M=[cntr1,cntr2,cntr3];
                if cntr3==10
                    M3='1_0';
                else
                    M3=num2str(M(3));
                end
                scatter3(Ps(:,M(1)).*L(M(1)),Ps(:,M(2)).*L(M(2)), ...
                    Ps(:,M(3)).*L(M(3)),20,A);
                title(['Diffusion sub-space: \psi_' num2str(M(1)) ...
                    '-\psi_' num2str(M(2)) '-\psi_' M3 ', color: ' Color]);
                colorbar
                pause(0.5);
            end
        end
    end
end

function Plot_Distances(S2,S2_diff)
    figure

    subplot(2,2,1)
    imagesc(S2)
    colorbar
    %daspect([1 1 1])
    caxis([0 1])
    title('Euclidean distances of 2D images')
    
    subplot(2,2,2)
    imagesc(S2_diff)
    colorbar
    %daspect([1 1 1])
    caxis([0 1])
    title('Diffusion distances of 2D images')

    subplot(2,2,3)
    spy(S2)
    %daspect([1 1 1])
    title('Euclidean distances of 2D images')
    
    subplot(2,2,4)
    spy(S2_diff)
    %daspect([1 1 1])
    title('Diffusion distances of 2D images')
end

%%  Decoding exit status of the optimization algorithm
function Exit_Status=Decode_Status(ExitFlag)
    switch ExitFlag
        case 1
            Exit_Status='Function converged to a solution x.';
        case 2
            Exit_Status='Change in x was less than the specified tolerance.';
        case 3
            Exit_Status='Change in the residual was less than the specified tolerance.';
        case 4
            Exit_Status='Magnitude of search direction was smaller than the specified tolerance.';
        case 0
            Exit_Status='Number of iterations or function evaluations exceeded the maximum.';
        case -1
            Exit_Status='Output function terminated the algorithm.';
        case -2
            Exit_Status='Problem is infeasible: the bounds lb and ub are inconsistent.';
        case -3
            Exit_Status='Regularization parameter became too large (levenberg-marquardt algorithm).';
        case -4
            Exit_Status='Line search could not sufficiently decrease the residual along the current search direction.';
    end
end    

%% Imposing the rotation matrix constraints to find {c}
function G_Functional=OR_Func(c,PsMod,Rotation)
    %Note: PsMod=Ps(:,1+(1:NPsi))'
    N_R=int8(3);
    r=size(PsMod,2);
    r_MP5=1/sqrt(r);
    switch size(PsMod,1)
        case 2
            N_c=int8(2);
        case 9
            N_c=int8(9);
    end
    N_c_2=int8(N_c^2);
    R_Dim=int8([N_R N_R]);
    c_Temp=reshape(c,[N_c N_c]);

    Temp=zeros(N_c,N_c);
    Temp2=zeros(N_c_2,1);
    I=eye(N_R);

    Input_Flag=0;
    G_Functional=zeros(r,1);
    if Input_Flag
        R_Big=c_Temp*PsMod;
        for cntr_l=1:r
            R=reshape(R_Big(:,cntr_l),R_Dim);
            Temp=R-Rotation.R(:,:,cntr_l);
            Temp2=Temp(:);
            %G_Functional(cntr_l)=Temp2'*Temp2;                         %L2^2
            G_Functional(cntr_l)=sqrt(Temp2'*Temp2)+abs((det(R)-1));    %L2
            %G_Functional(cntr_l)=sum(abs(Temp2));                      %L1
        end
    else
        R_Big=c_Temp*PsMod;
        if N_c==2
            Ones=ones(1,r);
            Zeros=zeros(1,r);
            R_Big=[R_Big(1,:);-R_Big(2,:);Zeros;...
                R_Big(2,:);R_Big(1,:);Zeros;...
                Zeros;Zeros;Ones];
            clear Ones;
            clear Zeros;
        end
        for cntr_l=1:r
            R=reshape(R_Big(:,cntr_l),R_Dim);
            Temp=R'*R-I;
            Temp2=Temp(:);
            %G_Functional(cntr_l)=Temp2'*Temp2+(det(R)-1)^2;     %L2^2
            G_Functional(cntr_l)=sqrt(Temp2'*Temp2)+abs(det(R)-1);%L2
            %G_Functional(cntr_l)=sum(abs(Temp(:)))+abs(det(R)-1);  %L1
        end
    end
    %G_Functional=G_Functional*r_MP5;
    G_Functional=sqrt(G_Functional)*r_MP5;
end

function R_det_res=cPsi2R(c_scale,c,Ps)
    r=size(Ps,1);
    R_det_res=zeros(r,1);
    N_R=int8(3);
    N_R_2=int8(N_R^2);
    c_Temp=reshape(c*c_scale,[N_R_2 N_R_2]);
    Ps=Ps(:,2:end)';
    for cntr_l=1:r
        R_det_res(cntr_l)=det(reshape(c_Temp*Ps(:,cntr_l),R_Dim));
    end
    R_det_res=R_det_res-1;
    R_det_res=(R_det_res'*R_det_res)/r;
end

%% Returning the rotation matrix R
function R_recon=OR_Func_R(c,PsMod,Param)
    r=size(PsMod,2);
    N_R=Param.Dim.NR;
    N_c=Param.Dim.Nc;

    R_Dim=int8([N_R N_R]);
    R_Dim_Singlet=int8([N_R N_R 1]);

    c_Temp=reshape(c,[N_c N_c]);
    R_Big=c_Temp*PsMod;
    R_recon=zeros(N_R,N_R,r);
    for cntr_l=1:r
        R=R_Big(:,cntr_l);
        switch N_c
            case 2
                R=[R(1),-R(2),0;R(2),R(1),0;0,0,1];
            case 9
                R=reshape(R,R_Dim);
        end
        R_recon(:,:,cntr_l)=reshape(Polar_Decompose(R),R_Dim_Singlet);
    end
end

function R_orth=Polar_Decompose(R)
        [U,~,V]=svd(R);
        R_orth=U*V';
end


%%  Saving best diffusion coordinate
function [Recon,Rotation]=Recon_Matrices(Ps,c,Param,Rotation)
    %Initialization
    Recon=struct;
    Recon.c=c;
    EV_Index=1:Param.Dim.NPsi;
    EV_Index_1=EV_Index+1;
    N_R=Param.Dim.NR;
    N_orient_f=length(Param.FilterIndex);

    %Retrieval of rotation vectors for filtered snapshots only
    %PsMod=;
    %Recon_R=OR_Func_R(c,Ps(:,2:10)');
    %clear PsMod;

    %Reconstructing filtered Ps vectors with original order
    Nc=Param.Dim.Nc;
    c_unfold_inv=pinv(reshape(c,[9 9]));
    Recon.Ps=zeros(N_orient_f,Param.Dim.NPsi);

    switch Nc
        case 2
            for cntr=1:N_orient_f
                Recon.Ps(cntr,EV_Index)=c_unfold_inv*reshape(Recon.R(1:Nc,1,cntr),[Nc 1]);
            end
        case 9
            for cntr=1:N_orient_f
                Recon.Ps(cntr,EV_Index)=c_unfold_inv*reshape(Recon.R(:,:,cntr),[Nc 1]);
            end
    end
    
    %Total reconstruction error of filtered Ps
    Recon.Ps_Error=norm(Ps(Param.FilterIndex,EV_Index_1)-Recon.Ps(:,EV_Index),'fro')/...
        norm(Ps(Param.FilterIndex,EV_Index_1),'fro');

    %Correlation between filtered Ps and Ps_Recon
    Recon.Ps_corr=zeros(Param.Dim.NPsi,1);
    for cntr=EV_Index
        Recon.Ps_corr(cntr)=CorrCoeff(Ps(Param.FilterIndex,cntr+1),Recon.Ps(:,cntr),'Ps');
    end
    
    %Alignment matrix for retrieved rotation matrices of filtered snapshots
    Recon.Axis=zeros(3,N_orient_f); %Axis=Theta*[nx;ny;nz]
    Recon.Q=zeros(4,N_orient_f);

    R1_raw=reshape(Recon.R(:,:,1),[N_R N_R]);
    if Param.TransposeR
        R1_raw=R1_raw';
    end
    Recon.Axis(:,1)=RotMat2Axis(R1_raw);
    Recon.Q(:,1)=Axis2Quat(Recon.Axis(:,1));
    Q1Conj=QConj(Recon.Q(:,1));
    
    %Aligning rotation matrices of filtered snapshots
    for cntr=1:N_orient_f
        Temp=reshape(Recon.R(:,:,cntr),[N_R N_R]);
        if Param.TransposeR
            Temp=Temp';
        end
        Recon.Axis(:,cntr)=RotMat2Axis(Temp);
        Recon.Q(:,cntr)=Axis2Quat(Recon.Axis(:,cntr));
    end
    Recon.Q=QProduct(Recon.Q,Q1Conj);
    Recon.Euler=Quat2Euler(Recon.Q);
    Recon.Wigner=Wigner_D(Recon.Euler);
    Recon.Axis=Quat2Axis(Recon.Q);
    [Recon.Axis_norm,Recon.Angles]=SplitAxisAngle(Recon.Axis);
    
    %Re-alignment of input orientations
    Rotation.Q=Rotation.Q(:,Param.FilterIndex);
    Q1Conj=QConj(Rotation.Q(:,1));
    Rotation.Q=QProduct(Rotation.Q,Q1Conj);
    Rotation.Euler=Quat2Euler(Rotation.Q);
    Rotation.Wigner=Wigner_D(Rotation.Euler);
    Rotation.Axis=Quat2Axis(Rotation.Q);
    [Rotation.Axis_norm,Rotation.Angles]=SplitAxisAngle(Rotation.Axis);
    
    Recon.Error.Index_MeanSQ=0;
    Recon.Error.Index_Median=0;
    Recon.Error.Index_IQR=0;
    Recon.Mutual_S3_Error=0;
    Recon.Ps_distr=0;
    
    return
    
    %THE main factor
    N_orient=size(Recon.Axis,2);
    Mutual_S3_Error=zeros(N_orient,N_orient-1);
    Cntr_Index=1:N_orient;
    for cntr1=Cntr_Index
        for cntr2=Cntr_Index(Cntr_Index ~= cntr1)
            TempA=real(acos(abs(Rotation.Q(:,cntr1)'*Rotation.Q(:,cntr2))));
            TempB=real(acos(abs(Recon.Q(:,cntr1)'*Recon.Q(:,cntr2))));
            Mutual_S3_Error(cntr1,cntr2)=TempA-TempB;
        end
    end
    Mutual_S3_Error=2*abs(Mutual_S3_Error);
    Recon.Error.Index_MeanSQ=sqrt(sum(Mutual_S3_Error(:).^2)/(N_orient^2-N_orient));
    Recon.Error.Index_Median=median(Mutual_S3_Error(:));
    Recon.Error.Index_IQR=iqr(Mutual_S3_Error(:));
    Recon.Mutual_S3_Error=Mutual_S3_Error;
    Recon.Ps_distr=Param.Ps_distr;

    
    %Recon.c_gold=lsqr(Ps(Param.FilterIndex,EV_Index_1),Rotation.Wigner);
    
end

%% Plotting the final orientation recovery results
function Plot_orientation_recovery(Ps,Lambda,Recon,Rotation,Param)
    Abs_Flag=1;
    N_R=Param.Dim.N;
    %Param.FilterIndex
    % Plotting the residuals and their correlation function
    figure
    subplot(211)
    plot(Recon.Residual)
    title(['Residuals with ResNorm=' num2str(Recon.ResNorm)])
    subplot(212)
    plot(real(xcov(Recon.Residual)))
    title('Covariance of Residual')

    Plot_EigenVector_Recon(Ps,Recon,Param);
    Plot_C_Matrix(Recon,Param);
    Plot_Geodesic_Error(Recon);
    
    % Plotting (original and) reconstructed axes (individual components)
    figure
    set(gca,'NextPlot','replacechildren');
    [Axis_recon,Angle_recon]=SplitAxisAngle(Recon.Axis);

    subplot(4,1,1)
    %plot(xcorr(Angle_recon,Rotation.Angles))
    bar(Angle_recon,'r')
    title(['Reconstructed Angle [0,\pi), correlation coefficient = ' ...
        num2str(fix(100*CorrCoeff(Angle_recon,Rotation.Angles,'Angles'))) '%'])
    hold on
    bar(Rotation.Angles,'b')
    hold off
    legend('Reconstructed','Input');
    for cntr=1:N_R
        Temp1=Rotation.Axis_norm(cntr,:);
        Temp2=Axis_recon(cntr,:);
        subplot(4,1,1+cntr)
        bar(Temp1,'r')
        hold on
        bar(Temp2,'b')
        hold off
        legend('Input','Reconstructed');
        title(['Direction cosine ' num2str(cntr) ' of rotation axes (norm), ' ...
            ' correlation coefficient = ' ...
            num2str(fix(100*CorrCoeff(Temp1,Temp2,['Axis ' num2str(cntr)]))) '%'])
    end
    
    figure
    set(gca,'NextPlot','replacechildren');
    subplot(2,1,1)
    Temp1=cos(Rotation.Angles);
    Temp2=cos(Angle_recon);
    [~,Index]=sort(abs(Temp1-Temp2),'ascend');
    Temp1=Temp1(Index);
    Temp2=Temp2(Index);
    Texts.X='Input cos(angles)';
    Texts.Y='Reconstructed cos(angles)';
    Texts.Title=['Reconstructed cos(Angle) [-1,1), correlation coefficient = ' ...
        num2str(fix(100*CorrCoeff(Temp1,Temp2,'Cos(Angles)'))) '%'];
    Plot_X_Y_Distribution(Temp1,Temp2,Texts)
    subplot(2,1,2)
    Temp1=Rotation.Angles;
    Temp2=Angle_recon;
    Temp1=Temp1(Index);
    Temp2=Temp2(Index);
    Texts.X='Input angles';
    Texts.Y='Reconstructed angles';
    Texts.Title=['Reconstructed Angle [0,\pi), correlation coefficient = ' ...
        num2str(fix(100*CorrCoeff(Temp1,Temp2,'Angles'))) '%'];
    Plot_X_Y_Distribution(Temp1,Temp2,Texts)

    figure
    set(gca,'NextPlot','replacechildren');
    subplot(2,2,1)
    Temp1=Rotation.Angles;
    Temp2=Angle_recon;
    if Abs_Flag
        Temp1=abs(Temp1);
        Temp2=abs(Temp2);
    end
    Texts.X='Input angles';
    Texts.Y='Reconstructed angles';
    Texts.Title=['Reconstructed Angle [0,\pi), correlation coefficient = ' ...
        num2str(fix(100*CorrCoeff(Temp1,Temp2,'Angles'))) '%'];
    Plot_X_Y_Distribution(Temp1,Temp2,Texts)
    for cntr=1:3
        Temp1=Rotation.Axis_norm(cntr,:);
        Temp2=Axis_recon(cntr,:);
        subplot(2,2,cntr+1)
        Texts.X='Input axis';
        Texts.Y='Reconstructed axis';
        Texts.Title=['Direction cosine ' num2str(cntr) ' of rotation axes (norm), ' ...
            ' correlation coefficient = ' ...
            num2str(fix(100*CorrCoeff(Temp1,Temp2,['Axis ' num2str(cntr)]))) '%'];
        Plot_X_Y_Distribution(Temp1,Temp2,Texts)
    end
    
    Plot_Quaternions(Recon,Rotation,Abs_Flag);
end

function Plot_Geodesic_Error(Recon)
    %Plotting the entire geodesic distances
    plot(Recon.Mutual_S3_Error(:))
    xlabel('Pair Index')
    ylabel('Quaternion Distance')
    title(['\Deltaq(RMS) = ' num2str(Recon.Error.Index_MeanSQ) ...
        '\Deltaq(Median) = ' num2str(Recon.Error.Index_Median) ...
        '\Deltaq(Interquartile) = ' num2str(Recon.Error.Index_IQR)])
end

function Plot_Quaternions(Recon,Rotation,Abs_Flag)
    % Plotting (original and) reconstructed quaternions
    figure
    set(gca,'NextPlot','replacechildren');
    for cntr=1:4
        Temp1=Rotation.Q(cntr,:);
        Temp2=Recon.Q(cntr,:);
        if Abs_Flag
            Temp1=abs(Temp1);
            Temp2=abs(Temp2);
        end
        subplot(4,1,cntr)
        bar(Temp1,'r')
        hold on
        bar(Temp2,'b')
        hold off
        legend('Original','Reconstructed')
        title(['Quanternion component ' num2str(cntr), ...
            ' correlation coefficient = ' ...
            num2str(fix(100*CorrCoeff(Temp1,Temp2,['Quaternion ' num2str(cntr)]))) '%'])
    end
    
    figure
    set(gca,'NextPlot','replacechildren');
    for cntr=1:4
        Temp1=Rotation.Q(cntr,:);
        Temp2=Recon.Q(cntr,:);
        if Abs_Flag
            Temp1=abs(Temp1);
            Temp2=abs(Temp2);
        end
        subplot(2,2,cntr)
        Texts.X='Input quaternion';
        Texts.Y='Reconstructed quaternion';
        Texts.Title=['Quanternion component ' num2str(cntr), ...
            ' correlation coefficient = ' ...
            num2str(fix(100*CorrCoeff(Temp1,Temp2,['Quaternion ' num2str(cntr)]))) '%'];
        Plot_X_Y_Distribution(Temp1,Temp2,Texts)
    end
end

function Plot_EigenVector_Recon(Ps,Recon,Param)
    % Plotting the input and reconstructed eigenvectors
    figure
    set(gca,'NextPlot','replacechildren');
    for cntr=1:Param.Dim.NPsi
        subplot(4,3,cntr);
        plot(Ps(Param.FilterIndex,cntr+1)','b*-');
        hold on
        if (cntr <= Param.Dim.NPsi)
            plot(Recon.Ps(:,cntr),'ro-');
            legend('Input \psi','Recon \psi')
        end
    end
    subplot(4,3,10)
    imagesc(Recon.Ps(:,1:Param.Dim.NPsi))
    title(['Reconstructed eigenvectors, Error= ' ...
        num2str(100*Recon.Ps_Error) '%']);
    subplot(4,3,11)
    surf(Ps(Param.FilterIndex,1+(1:Param.Dim.NPsi))); shading interp; 
    subplot(4,3,12)
    surf(Recon.Ps(:,1:Param.Dim.NPsi)); shading interp;
    
    %Plotting best and worst fits of eigenvectors
    [~,Index]=sort(Recon.Ps_corr,'descend');
    Index_Best=Index(1);
    Index_Worst=Index(end);
    figure
    set(gca,'NextPlot','replacechildren');

    subplot(221)
    Texts.X='Input eigenvector';
    Texts.Y='Reconstructed eigenvector';
    Texts.Title=['Best fit of eigenvectors, Correlation coefficient: ' ...
        num2str(Recon.Ps_corr(Index_Best))];
    Plot_X_Y_Distribution(Ps(Param.FilterIndex,1+Index_Best),Recon.Ps(:,Index_Best),Texts)

    subplot(222)
    Texts.X='Input eigenvector';
    Texts.Y='Reconstructed eigenvector';
    Texts.Title=['Worst fit of eigenvectors, Correlation coefficient: ' ...
        num2str(Recon.Ps_corr(Index_Worst))];
    Plot_X_Y_Distribution(Ps(Param.FilterIndex,1+Index_Worst),Recon.Ps(:,Index_Worst),Texts)
    
    subplot(223)
    bar((Ps(Param.FilterIndex,1+Index_Best)),'b');
    hold on
    bar((Recon.Ps(:,Index_Best)),'r');
    title(['Best fit of eigenvectors, Correlation coefficient: ' ...
        num2str(Recon.Ps_corr(Index_Best))])
    
    subplot(224)
    bar((Ps(Param.FilterIndex,1+Index_Worst)),'b');
    hold on
    bar((Recon.Ps(:,Index_Worst)),'r');
    title(['Worst fit of eigenvectors, Correlation coefficient: ' ...
        num2str(Recon.Ps_corr(Index_Worst))])
end

function Plot_C_Matrix(Recon,Param)
    % Plotting the {C} coefficients
    figure
    set(gca,'NextPlot','replacechildren');

    subplot(221)
    plot(sort(Recon.c(:)),'b*')
    Simulate_Flag=0;
    if Simulate_Flag
        hold on
        plot(sort(c_set(:)),'ro--')
        hold off
        legend('Reconstructed','Input');
        Error_c_recon=norm(c(:)-c_set(:))/norm(c_set(:));
        title(['Error in estimation of {c}=' num2str(100*Error_c_recon) '%'])
    end
    xlabel('C indices')
    ylabel('C values')
    title([Recon.Exit_Status ', ResNorm=' num2str(Recon.ResNorm)])
    
    N_c=Param.Dim.Nc;
    subplot(223)
    imagesc(reshape(Recon.c,[N_c N_c]))
    colorbar
    axis image;
    xlabel('Column indices')
    ylabel('Row indices')
    
    subplot(222)
    plot(sort(Recon.c_gold(:)),'b*')
    Simulate_Flag=0;
    if Simulate_Flag
        hold on
        plot(sort(c_set(:)),'ro--')
        hold off
        legend('Reconstructed','Input');
        Error_c_recon=norm(c(:)-c_set(:))/norm(c_set(:));
        title(['Error in estimation of {c}=' num2str(100*Error_c_recon) '%'])
    end
    xlabel('C indices')
    ylabel('C values')
    
    N_c=Param.Dim.Nc;
    subplot(224)
    imagesc(reshape(Recon.c_gold,[N_c N_c]))
    colorbar
    axis image;
    xlabel('Column indices')
    ylabel('Row indices')
end

function []=Plot_X_Y_Distribution(x,y,Texts)
    x=x(:);
    y=y(:);
    if sum(isnan(x))
        disp(['NaN in the first argument of Plot_X_Y: ' Texts.X]);
    elseif sum(isnan(y))
        disp(['NaN in the second argument of Plot_X_Y: ' Texts.Y]);
    elseif ~sum(abs(x))
        disp(['All-zero in the first argument of Plot_X_Y: ' Texts.X]);
    elseif ~sum(abs(y))
        disp(['All-zero in the second argument of Plot_X_Y: ' Texts.Y]);
    else
        z=[x;y];
        Min=min(z);
        Max=max(z);
        Color_Code=1;
        switch Color_Code
            case 1
                scatter(x,y,5,1:length(x));
            case 2
                scatter(x,y,5,x);
            case 3
                scatter(x,y,5,y);
            case 4
                scatter(x,y,5,sort(abs(x-y),'ascend'));
        end
        hold on;
        plot([Min Max],[Min Max],'--r');
        colorbar
        hold off;
        xlim([Min Max]);
        ylim([Min Max]);
        daspect([1 1 1]);
        title(Texts.Title)
        xlabel(Texts.X);
        ylabel(Texts.Y);
    end
end

function Plot_Metric_Map(Metrics)
    Metrics=Metrics(:);
    N=size(Metrics,1);
    for cntr=1:N
        Temp=Metrics{cntr};
        Epsilon(cntr)=Temp{1};
        d(cntr)=Temp{2};
        ResNorm(cntr)=Temp{3};
        CorrCoeffMin(cntr)=Temp{4};
        CorrCoeffMax(cntr)=Temp{5};
        EVRecPercError(cntr)=Temp{6};
        RMSAngPercError(cntr)=Temp{7};
    end
    
    figure
    set(gca,'NextPlot','replacechildren');
    subplot(321)
    plot(Epsilon)
    title('\epsilon')
    subplot(322)
    plot(d)
    title('d')
    subplot(323)
    plot(ResNorm)
    title('ResNorm')
    subplot(324)
    plot(CorrCoeffMin,'b')
    hold on
    plot(CorrCoeffMax,'r')
    hold off
    legend('CorrCoeffMin','CorrCoeffMax')
    subplot(325)
    plot(EVRecPercError)
    title('EVRecPercError')
    subplot(326)
    plot(RMSAngPercError)
    title('RMSAngPercError')

    figure
    set(gca,'NextPlot','replacechildren');
    [~,Index]=sort(RMSAngPercError,'ascend');
    subplot(321)
    plot(Epsilon(Index))
    title('\epsilon')
    subplot(322)
    plot(d(Index))
    title('d')
    subplot(323)
    plot(ResNorm(Index))
    title('ResNorm')
    subplot(324)
    plot(CorrCoeffMin(Index),'b')
    hold on
    plot(CorrCoeffMax(Index),'r')
    hold off
    legend('CorrCoeffMin','CorrCoeffMax')
    subplot(325)
    plot(EVRecPercError(Index))
    title('EVRecPercError')
    subplot(326)
    plot(RMSAngPercError(Index))
    title('RMSAngPercError (sorted)')

    figure
    set(gca,'NextPlot','replacechildren');
    [~,Index]=sort(ResNorm,'ascend');
    subplot(321)
    plot(Epsilon(Index))
    title('\epsilon')
    subplot(322)
    plot(d(Index))
    title('d')
    subplot(323)
    plot(ResNorm(Index))
    title('ResNorm (sorted)')
    subplot(324)
    plot(CorrCoeffMin(Index),'b')
    hold on
    plot(CorrCoeffMax(Index),'r')
    hold off
    legend('CorrCoeffMin','CorrCoeffMax')
    subplot(325)
    plot(EVRecPercError(Index))
    title('EVRecPercError')
    subplot(326)
    plot(RMSAngPercError(Index))
    title('RMSAngPercError')
    
    figure
    set(gca,'NextPlot','replacechildren');
    [~,Index]=sort(EVRecPercError,'ascend');
    subplot(321)
    plot(Epsilon(Index))
    title('\epsilon')
    subplot(322)
    plot(d(Index))
    title('d')
    subplot(323)
    plot(ResNorm(Index))
    title('ResNorm')
    subplot(324)
    plot(CorrCoeffMin(Index),'b')
    hold on
    plot(CorrCoeffMax(Index),'r')
    hold off
    legend('CorrCoeffMin','CorrCoeffMax')
    subplot(325)
    plot(EVRecPercError(Index))
    title('EVRecPercError (sorted)')
    subplot(326)
    plot(RMSAngPercError(Index))
    title('RMSAngPercError')
    
    figure
    set(gca,'NextPlot','replacechildren');
    [~,Index]=sort(CorrCoeffMin,'descend');
    subplot(321)
    plot(Epsilon(Index))
    title('\epsilon')
    subplot(322)
    plot(d(Index))
    title('d')
    subplot(323)
    plot(ResNorm(Index))
    title('ResNorm')
    subplot(324)
    plot(CorrCoeffMin(Index),'b')
    hold on
    plot(CorrCoeffMax(Index),'r')
    hold off
    legend('CorrCoeffMin (sorted)','CorrCoeffMax')
    subplot(325)
    plot(EVRecPercError(Index))
    title('EVRecPercError')
    subplot(326)
    plot(RMSAngPercError(Index))
    title('RMSAngPercError')
    
    figure
    set(gca,'NextPlot','replacechildren');
    [~,Index]=sort(CorrCoeffMax,'descend');
    subplot(321)
    plot(Epsilon(Index))
    title('\epsilon')
    subplot(322)
    plot(d(Index))
    title('d')
    subplot(323)
    plot(ResNorm(Index))
    title('ResNorm')
    subplot(324)
    plot(CorrCoeffMin(Index),'b')
    hold on
    plot(CorrCoeffMax(Index),'r')
    hold off
    legend('CorrCoeffMin','CorrCoeffMax (sorted)')
    subplot(325)
    plot(EVRecPercError(Index))
    title('EVRecPercError')
    subplot(326)
    plot(RMSAngPercError(Index))
    title('RMSAngPercError')
    
    figure
    set(gca,'NextPlot','replacechildren');
    subplot(321)
    SurfPlot(Epsilon,d,Epsilon)
    title('\epsilon')
    subplot(322)
    SurfPlot(Epsilon,d,d)
    title('d')
    subplot(323)
    SurfPlot(Epsilon,d,ResNorm)
    title('ResNorm')
    subplot(626)
    SurfPlot(Epsilon,d,CorrCoeffMin)
    title('CorrCoeffMin')
    subplot(628)
    SurfPlot(Epsilon,d,CorrCoeffMax)
    title('CorrCoeffMax')
    subplot(325)
    SurfPlot(Epsilon,d,EVRecPercError)
    title('EVRecPercError')
    subplot(326)
    SurfPlot(Epsilon,d,RMSAngPercError)
    title('RMSAngPercError')
end

function SurfPlot(a,b,x)
    a=a(:);
    A=Grid_1D(a);
    if size(A,1) < size(a,1)
        B=reshape(b(:),size(A));
        X=reshape(x(:),size(A));
        %imagesc(X);
        surf(A,B,X);
        shading interp;
        view(0,90);
        colorbar
        axis tight
        axis square
    else
        title('No 2D maps w/ a constant parameter');
    end
end

function a_=Grid_1D(a)
    a=a(:);
    N=length(a);
    N1=1;
    x=a(1);
    cntr=2;
    while (cntr <= N)
        if (a(cntr) ~= a(cntr-1)) && all(x-a(cntr))
            N1=N1+1;
            x=[x;a(cntr)];
        end
        cntr=cntr+1;
    end
    N2=N/N1;
    a_=reshape(a,[N1 N2]);
    %if a(2)~=a(1)
    %    a_=a_';
    %end
end

%% General rotation conversion functions
function W=Wigner_D(Euler)
    %Euler=[Phi;Theta;Psi]
    %Phi, Psi: [0,2pi]; Theta:[0,pi]
    
    Psi=Euler(1,:);
    Theta=Euler(2,:);
    Phi=Euler(3,:);

    Sin_=sin(Theta);
    Cos_=cos(Theta);

    W=[...
        Cos_;...
    cos(Psi).*Sin_;...
    sin(Psi).*Sin_;...
    cos(Phi).*Sin_;...
    sin(Phi).*Sin_;...
    sin(Phi+Psi).*(1+Cos_)/sqrt(2);...
    cos(Phi+Psi).*(1+Cos_)/sqrt(2);...
    sin(Psi-Phi).*(1-Cos_)/sqrt(2);...
    cos(Psi-Phi).*(1-Cos_)/sqrt(2);...
    ];
    
%    W.D1=cos(Theta);
%    W.D2=cos(Phi).*sin(Theta)/sqrt(2);
%    W.D3=sin(Phi).*sin(Theta)/sqrt(2);
%    W.D4=sin(Theta).*cos(Psi)/sqrt(2);
%    W.D5=sin(Theta).*sin(Psi)/sqrt(2);
%    W.D6=sin(Phi+Psi).*cos(Theta/2).^2;
%    W.D7=cos(Phi+Psi).*cos(Theta/2).^2;
%    W.D8=sin(Phi-Psi).*sin(Theta/2).^2;
%    W.D9=cos(Phi-Psi).*sin(Theta/2).^2;
end

function Rotation=Load_Rotation(Axis_File,Param)
    Rotation=struct;
    Temp=load(Axis_File);
    Rotation.Axis=Temp.Axis;
    %Rotation.R=Axis2RotMatArray(Rotation.Axis);
    Rotation.R=Axis2RotUnfold(Rotation.Axis);
%    [Rotation.R,Rotation.Axis]=Align_Rotations(Temp.Axis,Param);
    Rotation.Q=Axis2Quat(Rotation.Axis);
    Rotation.Euler=Quat2Euler(Rotation.Q);
    Rotation.Wigner=Wigner_D(Rotation.Euler);
    [Rotation.Axis_norm,Rotation.Angles]=SplitAxisAngle(Rotation.Axis);
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

function R=Axis2RotMatArray(Axis)
    N_R=3;
    N_orient=size(Axis,2);
    R=zeros(N_R,N_R,N_orient);
    for cntr=1:N_orient
        R(:,:,cntr)=Axis2RotMat(Axis(:,cntr));
    end
end

function [R,Axis_new]=Align_Rotations(Axis,Param)
    N_R=Param.Dim.NR;
    N_orient=size(Axis,2);
    R1=Axis2RotMat(Axis(:,1));
    if Param.TransposeR
        R1=R1';
    end
    Alignment=R1';
    Axis_new=zeros(size(Axis));
    R=zeros(N_R,N_R,N_orient);
    for cntr=1:N_orient
        Temp=Axis2RotMat(Axis(:,cntr));
        if Param.TransposeR
            Temp=Temp';
        end
        %Temp=Polar_Decompose(Temp*Alignment);
        R(:,:,cntr)=Temp;
        Axis_new(:,cntr)=RotMat2Axis(Temp);
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
    %switch size(R,1)
        %case 2
            %Axis=[0;0;0];
            %Axis(3)=atan2(R(2,1),R(1,1));
        %case 3
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
    %end
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

function Euler=WignerD2Euler(WignerD)
    Norm=repmat(sqrt(sum(WignerD.^2,1)),9,1);
    WignerD=WignerD.*(sqrt(3)./Norm);
    
    Psi=atan2(WignerD(3,:),WignerD(2,:));
    Phi=atan2(WignerD(5,:),WignerD(4,:));
    Theta=real(acos(WignerD(1,:)));
    
    Euler=[Psi;Theta;Phi];
end

function Euler=Quat2Euler(Q)
    if size(Q,1)>4
        Flip=1;
        Q=Q';
    else
        Flip=0;
    end
    Method=1;
    switch Method
        case 1  %First and good implementation
            Phi=atan2(Q(1,:).*Q(3,:)+Q(2,:).*Q(4,:),Q(2,:).*Q(3,:)-Q(1,:).*Q(4,:));
            Theta=real(acos(-Q(1,:).^2-Q(2,:).^2+Q(3,:).^2+Q(4,:).^2));
            Psi=-atan2(Q(1,:).*Q(3,:)-Q(2,:).*Q(4,:),Q(2,:).*Q(3,:)+Q(1,:).*Q(4,:));
        case 2
            Phi=atan2(Q(1,:).*Q(3,:)+Q(2,:).*Q(4,:),Q(2,:).*Q(3,:)-Q(1,:).*Q(4,:));
            Theta=acos(-Q(1,:).^2-Q(2,:).^2+Q(3,:).^2+Q(4,:).^2);
            Psi=-atan2(Q(1,:).*Q(3,:)-Q(2,:).*Q(4,:),Q(2,:).*Q(3,:)+Q(1,:).*Q(4,:));
    end
    Euler=[Phi;Theta;Psi];
    if Flip
        Euler=Euler';
    end
end

function [Axis_norm,Angle]=SplitAxisAngle(Axis)
    if (size(Axis,1) > 3)
        Axis=Axis';
    end
    N=size(Axis,2);
    Angle=zeros(1,N);
    Axis_norm=zeros(3,N);
    for cntr=1:N
        Angle(cntr)=norm(Axis(:,cntr));
        if Angle(cntr) ~= 0
            Axis_norm(:,cntr)=Axis(:,cntr)/Angle(cntr);
        end
    end
end

function Q_out=QConj(Q_in)
    Q_out=[Q_in(1);-Q_in(2:4)];
end

function Q=QProduct(QA,QB)
    N=size(QA,2);
    Q=zeros(4,N);
    for cntr=1:N
        Q(:,cntr)=QProductSingle(QA(:,cntr),QB);
    end
end

function Q=QProductSingle(QA,QB)
    Q=zeros(4,1);
    Q(1)=QA(1)*QB(1)-QA(2:4)'*QB(2:4);
    Q(2:4)=QA(1)*QB(2:4)+QB(1)*QA(2:4)+cross(QA(2:4),QB(2:4));
    Q=Q/norm(Q);
end

%% Miscellaneous functions
% Correlation coefficient
function r=CorrCoeff(x,y,Text)
    x=x(:);
    y=y(:);
    if sum(isnan(x))
        disp(['NaN in the first argument of correlation, ',Text])
        r=2011;
    elseif sum(isnan(y))
        disp(['NaN in the second argument of correlation, ',Text])
        r=2012;
    elseif ~sum(abs(x))
        disp(['All-zero in the first argument of correlation, ',Text])
        r=2013;
    elseif ~sum(abs(y))
        disp(['All-zero in the second argument of correlation, ',Text])
        r=2014;
    else
        x=x-mean(x);
        y=y-mean(y);
        r=x'*y/sqrt((x'*x)*(y'*y));
    end
end

% Logspace in an easy way!
function Geometric_Prog=logspace_2(a,b,n)
    k=(b/a)^(1/(n-1));
    Geometric_Prog=a.*k.^linspace(0,n-1,n);
end

%% Generating snapshots
function []= Rotated_Images()
    % Input parameters
    N_loop=28^3;   %Cube of an integer
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
    R=RandomRotMatrices(N_loop);
    
    R_size=[3 3];
    WaitBar=waitbar(0,WaitBar,'Memory allocation');drawnow;
    Images=randn(N_p2,N_loop);  %Memory allocation
    WaitBar=waitbar(0,WaitBar,['Generating ' num2str(N_loop) ' snapshots']);drawnow;

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
            waitbar(cntr/N_loop,WaitBar);drawnow;
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


% Loading protein
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
    %F=abs(1-(A.^2+B.^2+C.^2).*(1-1.5*(A-0.4).*(B-0.2).*(C+0.1)));
    %F( (A.^2+B.^2+C.^2 > 1) | (abs(sin(1.6*((A+0.2).^2+0.8*(B+0.1).*(C-0.2)+pi/12))) > 0.5) )=0;
    F=(1-0.4*((A-0.15).^2+(B+0.2).^2+(C-0.1).*(A-0.15).*(B+0.2)));
    %F( (((A-0.1).^2+((B+0.05)-0.8*(C-0.2)).^2 <=1) | (((A-0.2).*sin(B-0.3).*exp(-2*(C+0.5))) > 0.2)) & (x.^2+y.^2+z.^2<0.25))=1;
    F(  (cos(20*pi*(x-z-0.2).*abs(y+z+0.1).*abs(z-0.3)) < 0.2) | (A.^2+B.^2+C.^2 >1)  | (F<0) )=0;
    %F=F.*(1-(x+0.1).*(y-0.1).*(z-0.1)/6);
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

% k-space

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
%    disp(['Nyquist.x=' num2str(max(Nyquist.x(:)))])
%    disp(['Nyquist.z=' num2str(max(Nyquist.z(:)))])
end

function [Nyquist,k]=FourierScaledAxis(Number,Length)
    d=Length/(Number-1);
    Nyquist=0.5/d;

    %k=((1:Number)-1)*(2*Nyquist/Number);
    %Index=(k > Nyquist);
    %k(Index)=k(Index)-(2*Nyquist);

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


% Rotations
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
    
    %Index=randperm(n1^3);
    Index=1:N_orient;
    Psi=Psi(Index);
    Theta=Theta(Index);
    Phi=Phi(Index);
        
    Hopf_Switch=0;
    if Hopf_Switch
        Q=[cos(Theta/2).*cos(Psi/2),...
          cos(Theta/2).*sin(Psi/2),...
          sin(Theta/2).*cos(Phi+Psi/2),...
          sin(Theta/2).*sin(Phi+Psi/2)]';
    else
        Q=[cos(Theta/2).*cos(Psi/2),...
          sin(Theta/2).*sin(Phi+Psi/2),...
          sin(Theta/2).*cos(Phi+Psi/2),...
          cos(Theta/2).*sin(Psi/2)]';
    end
    
    Q=UnitMagPos(Q);
    SO3=struct;
    SO3.Q=Q;
    %[SO3.S2_Q,SO3.N_Q,SO3.dQ_min]=SortedDistance(Q);
    %save SO3.mat SO3
    pause(1e-6);
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


function R=RandomRotMatrices(N)
    Mode=4;
    switch Mode
        case 1
            x=zeros(1,N);
            y=x;
            Theta=(pi)*rand(1,N);
            z=Theta.*cos(Theta);
            Axis=[x;y;z];
        case 2
            Temp=([1;1]/N)*(1:N);
            CosTheta=2*Temp(1,:)-1;
            Theta=acos(CosTheta);
            Phi=(2*pi)*Temp(2,:);
            Sin=Theta.*sin(Theta);
            x=Sin.*cos(Phi);
            y=Sin.*sin(Phi);
            z=Theta.*cos(Theta);
            Axis=[x;y;z];
        case 3
            Q=Uniform_SO3_PDF(N);
            Axis=Quat2Axis(Q);
        case 4
            Q=Uniform_SO3_Hopf(N);           
            Axis=Quat2Axis(Q);
    end
    R=Axis2RotMatBatch(Axis);
    save Axis.mat Axis  '-v7.3';
end

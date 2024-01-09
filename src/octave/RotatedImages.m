function []= RotatedImages()
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
    save '../../artifacts/Images.mat' Images '-v6';
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

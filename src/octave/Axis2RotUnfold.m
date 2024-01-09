function R=Axis2RotUnfold(Axis)
    N_R2=9;
    N_orient=size(Axis,2);
    R=zeros(N_R2,N_orient);
    for cntr=1:N_orient
        Temp=Axis2RotMat(Axis(1:3,cntr));
        R(1:N_R2,cntr)=Temp(:);
    end
end


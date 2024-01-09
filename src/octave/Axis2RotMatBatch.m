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


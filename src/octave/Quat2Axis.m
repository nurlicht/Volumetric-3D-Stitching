function Axis=Quat2Axis(Q)
    N=size(Q,2);
    Axis=zeros(3,N);
    for cntr=1:N
        Axis(:,cntr)=Quat2AxisSingle(Q(:,cntr));
    end
end


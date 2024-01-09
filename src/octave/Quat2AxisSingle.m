function Axis=Quat2AxisSingle(Q)
    Angle=real(2*acos(abs(Q(1))));
    if ~Angle
        Axis=zeros(3,1);
    else
        Axis_norm=Q(2:4)/norm(Q(2:4));
        Axis=Angle*Axis_norm;
    end
end


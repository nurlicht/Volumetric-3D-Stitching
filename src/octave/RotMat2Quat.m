function Q=RotMat2Quat(R)
    Axis=RotMat2Axis(R);
    Q=Axis2Quat(Axis);
end


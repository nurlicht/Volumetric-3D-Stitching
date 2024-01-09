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


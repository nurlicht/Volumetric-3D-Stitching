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


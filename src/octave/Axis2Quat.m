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


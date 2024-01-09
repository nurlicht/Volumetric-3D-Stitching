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


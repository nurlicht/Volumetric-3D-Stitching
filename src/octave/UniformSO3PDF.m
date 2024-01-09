function Q=UniformSO3PDF(N)
    Q=randn(4,N);
    Norm=sqrt(Q(1,:).^2+Q(2,:).^2+Q(3,:).^2+Q(4,:).^2);
    for cntr=1:4
        Q(cntr,:)=Q(cntr,:)./Norm;
    end
end


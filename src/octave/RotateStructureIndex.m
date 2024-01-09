function Protein=RotateStructureIndex(F,R)
    N=max(size(F));
    [x,y,z]=meshgrid(((1:N)-(N+1)/2)/(N/2));
    Q=R*[x(:),y(:),z(:)]';
    Qx=reshape(Q(1,:),[N N N]);
    Qy=reshape(Q(2,:),[N N N]);
    Qz=reshape(Q(3,:),[N N N]);
    Protein=interp3(x,y,z,F,Qx,Qy,Qz,'linear',0);
end


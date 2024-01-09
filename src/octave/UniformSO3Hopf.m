function Q=UniformSO3Hopf(N_orient)
    n1=round(N_orient^(1/3));
    N_orient=n1^3;
    N=[n1 n1 n1];

    Psi_=(2*pi)*linspace(0,1,N(1)+1);
    %Theta_=acos(Factor*linspace(1,-1,N(2)+1));
    Theta_=acos(linspace(1,-1,N(2)));
    Phi_=(2*pi)*linspace(0,1,N(3)+1);

    Psi_=Psi_(1:N(1));
    Theta_=Theta_(1:N(2));
    Phi_=Phi_(1:N(3));

    Psi_=Psi_-mean(Psi_);
    Theta_=Theta_-mean(Theta_)+(pi/2);
    Phi_=Phi_-mean(Phi_);

    [Psi,Theta,Phi]=meshgrid(Psi_,Theta_,Phi_);
    Psi=Psi(:);
    Theta=Theta(:);
    Phi=Phi(:);

    Index=1:N_orient;
    Psi=Psi(Index);
    Theta=Theta(Index);
    Phi=Phi(Index);

    Q=[cos(Theta/2).*cos(Psi/2),...
      sin(Theta/2).*sin(Phi+Psi/2),...
      sin(Theta/2).*cos(Phi+Psi/2),...
      cos(Theta/2).*sin(Psi/2)]';
    Q=UnitMagPos(Q);
end


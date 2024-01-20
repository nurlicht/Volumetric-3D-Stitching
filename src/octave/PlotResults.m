function PlotResults(Ps,Lambda,Rotation)
    Plot_Eigenvalue_Eigenvector(Ps,Lambda)
    Plot_Individual_EV(Ps,Lambda)
    Plot_Diffusion_Statistics(Ps)
    Plot_Corr_Ps(Ps);
    Plot_Psi234_ColorRot(Ps,Lambda,Rotation,'Axis')
end
function Plot_Eigenvalue_Eigenvector(Ps,Lambda)
    figure
    set(gca,'NextPlot','replacechildren');
    subplot(211)
    imagesc(Ps)
    colorbar
    title('Diffusion map eigenfunctions {\psi_i}')
    subplot(212)
    plot(Lambda,'-*')
    colorbar
    title('Diffusion map eigenvalues')
end
function Plot_Diffusion_Statistics(Ps)
    figure
    set(gca,'NextPlot','replacechildren');
    subplot(221)
    bar(Ps(:));
    title(['First 10 \psi, min=' num2str(min(Ps(:))) ...
        ', Max=' num2str(max(Ps(:)))]);
    PsNorm=Ps(:,1).^2;
    for cntr=2:10
        PsNorm=PsNorm+Ps(:,cntr).^2;
    end
    PsNorm=sqrt(PsNorm);
    subplot(222)
    bar(PsNorm);
    title(['10-element norm, min=' num2str(min(PsNorm)) ...
        ', Max=' num2str(max(PsNorm))]);
    subplot(223)
    histfit(PsNorm,500);
    title('10-element norm histogram');
    subplot(224)
    Index=(1:size(PsNorm,1))';
    Index=2*pi*Index/max(Index);
    polar(Index,PsNorm,'b');
    title('10-element norm polar histogram');
    hold on
    polar(Index,mean(PsNorm)*ones(size(PsNorm)),'r--');
    hold off
end
function Plot_Individual_EV(Ps,Lambda)
    figure
    set(gca,'NextPlot','replacechildren');
    for cntr=1:9
        subplot(3,3,cntr)
        plot(Ps(:,cntr+1).*Lambda(cntr+1))
        title(['\psi_' num2str(cntr+1)])
    end
end
function Plot_Psi234_ColorRot(Ps,Lambda,Rot,Text)
    for cntr=1:size(Rot,1)
        figure
        set(gca,'NextPlot','replacechildren');

        subplot(221)
        scatter(Ps(:,2).*Lambda(2),Ps(:,3).*Lambda(3),20,Rot(cntr,:))
        title(['Phase plane: \psi_2 vs. \psi_3 - Color-coded by ' ...
            Text ' ' num2str(cntr)]);
        colorbar

        subplot(222)
        scatter(Ps(:,2).*Lambda(2),Ps(:,4).*Lambda(4),20,Rot(cntr,:))
        title(['Phase plane: \psi_2 vs. \psi_4 - Color-coded by ' ...
            Text ' ' num2str(cntr)]);
        colorbar

        subplot(223)
        scatter(Ps(:,3).*Lambda(3),Ps(:,4).*Lambda(4),20,Rot(cntr,:))
        title(['Phase plane: \psi_3 vs. \psi_4 - Color-coded by ' ...
            Text ' ' num2str(cntr)]);
        colorbar

        subplot(224)
        scatter3(Ps(:,2).*Lambda(2),Ps(:,3).*Lambda(3),...
            Ps(:,4).*Lambda(4),20,Rot(cntr,:))
        title(['Phase plane: \psi_2-\psi_3-\psi_4 - Color-coded by ' ...
            Text ' ' num2str(cntr)]);
        colorbar
    end
    drawnow;
end
function Plot_Corr_Ps(Ps)
    figure
    subplot(4,1,1)
    plot(real(xcov(Ps(:,2),Ps(:,2))))
    title('Covariance of \psi_2 and \psi_2');
    legend('Auto correlation')
    ylim([-0.2 1.1])
    subplot(4,1,2)
    plot(real(xcov(Ps(:,2),Ps(:,3))))
    title('Covariance of \psi_2 and \psi_3');
    legend('Cross correlation')
    ylim([-0.2 1.1])
    subplot(4,1,3)
    plot(real(xcov(Ps(:,2),Ps(:,4))))
    title('Covariance of \psi_2 and \psi_4');
    legend('Cross correlation')
    ylim([-0.2 1.1])
    subplot(4,1,4)
    plot(real(xcov(Ps(:,3),Ps(:,4))))
    title('Covariance of \psi_3 and \psi_4');
    legend('Cross correlation')
    ylim([-0.2 1.1])
    drawnow;
end

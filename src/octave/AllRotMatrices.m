function [R, Axis]=AllRotMatrices(N)
    Mode=2;
    switch Mode
        case 1  %Random
            Q=UniformSO3PDF(N);
            Axis=Quat2Axis(Q);
        case 2  %Hopf
            Q=UniformSO3Hopf(N);
            Axis=Quat2Axis(Q);
    end
    R=Axis2RotMatBatch(Axis);
    save '../../artifacts/Q.mat' Q '-text';
    save '../../artifacts/Axis.mat' Axis '-text';
    save '../../artifacts/R.mat' R '-text';
end


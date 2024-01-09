%% Major functions
function []=DiffusionMap3dOrientations(taskIndex)

    %Initialization
    close all force;
    %clear all;
    clc;

    %Similarity Matrix parameters
    k=150;
    Epsilon=0.7;

    %Whether to use known rotation matrices to estimate c(9,9)
    aPrioriFlag=1;

    if (~exist('taskIndex'))
      taskIndex = menu('Please select the operation',...
          'Generation of snapshots',...
          'Calculation of Distance Matrix',...
          'Estimation of Diffusion Coordinate');
    end

    %Main menu
    switch taskIndex;
        case 1
            RotatedImages();
        case 2
            DistanceMatrix(k);
            disp(['k=' num2str(k)]);
        case 3
            CalculateDiffusion(k,Epsilon,aPrioriFlag);
            disp(['k=' num2str(k) ', Epsilon=' num2str(Epsilon)]);
    end
end


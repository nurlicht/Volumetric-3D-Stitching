function []=DiffusionMap3dOrientations(args)

    %Initialization
    close all force;
    clc;
    if (~exist('args'))
      clear all;
      args = struct('nRot1D', 6, 'k', 150, 'Epsilon', 0.7, 'aPrioriFlag', 1);
    end
    save '../../artifacts/args.mat' args -text;

    %Whether to use known rotation matrices to estimate c(9,9)
    %aPrioriFlag=args.aPrioriFlag; %1

    [Axis, Images] = RotatedImages(args.nRot1D);
    [S2, N] = DistanceMatrix(Images, args.k);
    clear Images;
    disp(['k=' num2str(args.k)]);
    save '../../artifacts/S2.mat' S2 -text;
    save '../../artifacts/N.mat' N -text;
    Rotation_Axis=importdata('../../artifacts/Axis.mat');
    [Ps, Lambda, c0, c] = CalculateDiffusion(S2, N, Rotation_Axis, args.k,args.Epsilon,args.aPrioriFlag);
    disp(['k=' num2str(args.k) ', Epsilon=' num2str(args.Epsilon)]);
    save '../../artifacts/Ps.mat' Ps -text;
    save '../../artifacts/Lambda.mat' Lambda -text;
    save '../../artifacts/c0.mat' c0 -text;
    save '../../artifacts/c.mat' c -text;
end


function R_orth=PolarDecompose(R)
    [U,~,V]=svd(R);
    R_orth=U*V';
end

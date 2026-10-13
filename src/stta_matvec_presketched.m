function w = stta_matvec_presketched(A, s, p, Y, v)
%STTA_MATVEC_PRESKETCHED  One pre-sketched product, preparation included.
%
%   w = stta_matvec_presketched(A, s, p, Y, v)
%
%   Builds the sketched operator and the operator layouts with
%   stta_prepare_presketched, then performs the product with
%   stta_compress_presketched and recovers the tensor.  Everything
%   happens inside this call, so timing it charges the preparation to the
%   product -- the same accounting as every other route in the experiments.

P = stta_prepare_presketched(A, s, p, Y);
[Om, Ps] = stta_compress_presketched(P, v);
w = stta_recover(Om, Ps);
end
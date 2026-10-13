function A = rand_tt_operator(d, m, n, r)
%RAND_TT_OPERATOR  A random TT operator with dense cores of TT rank r.
%
%   A = rand_tt_operator(d, m, n, r) returns a d-dimensional tt_matrix whose
%   cores are [r, m, n, r] Gaussian arrays, scaled so that ||A|| stays O(1)
%   as d, n and r grow.  The cores are dense: nothing about the operator is
%   sparse or separable, which is the setting in which the m*n terms of the
%   cost dominate.

cr = cell(d,1);
rr = [1, r*ones(1,d-1), 1];
for k = 1:d
    cr{k} = randn(rr(k), m, n, rr(k+1)) / sqrt(rr(k)*rr(k+1)*n);
end
A = cell2core(tt_matrix, cr);
end
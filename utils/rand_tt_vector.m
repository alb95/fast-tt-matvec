function v = rand_tt_vector(d, n, r)
%RAND_TT_VECTOR  A random TT tensor of TT rank r.
%
%   v = rand_tt_vector(d, n, r) returns a d-dimensional tt_tensor whose cores
%   are [r, n, r] Gaussian arrays.  Its singular values do not decay, so the
%   rank r is genuine: it is the rank the algorithms have to carry.

cr = cell(d,1);
rr = [1, r*ones(1,d-1), 1];
for k = 1:d
    cr{k} = randn(rr(k), n, rr(k+1)) / sqrt(rr(k)*rr(k+1));
end
v = cell2core(tt_tensor, cr);
end
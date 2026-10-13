function v = rand_tt_vector_decay(d, n, r, decades, ref)
%RAND_TT_VECTOR_DECAY  A TT tensor of rank r whose singular values decay.
%
%   v = rand_tt_vector_decay(d, n, r, decades)
%   v = rand_tt_vector_decay(d, n, r, decades, ref)
%
%   The tensor is a sum of r random separable terms with weights spread
%   geometrically over DECADES decades, reaching 10^-decades at term REF
%   (default r, so the decay is spread over the whole rank),
%
%       v = sum_{j=1}^{r} 10^{-decades (j-1)/(ref-1)} u_j1 (x) ... (x) u_jd,
%
%   REF IS WHAT MAKES THE ACCURACY COLUMNS READABLE.  With the default the
%   weights fall very slowly at first -- 10^{-8/39} = 0.63 from one term to
%   the next at r = 40, decades = 8 -- so a rank-8 approximation of A*v, whose
%   own rank is r^A r = 160, cannot be good: the best one is off by 0.38, and
%   every route sits at or above that.  Nothing is wrong, but the column says
%   nothing either.  Setting REF near the target rank, say 2s, puts the decay
%   where the truncation happens, gives a floor around 10^-3, and lets the
%   routes be told apart.  It does not make v degenerate: the terms beyond REF
%   are small but present, and the TT rank is still r, which is all any of
%   the timings depend on.
%
%   so its TT ranks are r -- generically exactly r, since random separable
%   terms are independent -- while the singular values of every unfolding
%   follow the weights: random vectors in n dimensions are nearly
%   orthogonal, so the terms are nearly the singular directions themselves.
%
%   WHY NOT A TENSOR WITH RANDOM CORES.  Gaussian cores give a tensor whose
%   singular values are flat, and then no rank-s approximation is any good:
%   the best one is off by an amount comparable to the tensor itself, and a
%   sketched approximation, which is not a projection, can be off by more
%   than that.  A relative error of twenty is not a bug in that case, it is
%   the honest answer to a question nobody should ask.  It makes the
%   accuracy columns unreadable and it leaves the recovery working on a
%   matrix with no gap, which is not the regime any of these methods is for.
%
%   The timings do not care: the cost of every route depends on the RANKS,
%   which are r either way, and never on the values in the cores.

if nargin < 4 || isempty(decades), decades = 8; end
if nargin < 5 || isempty(ref),     ref     = r; end

w = 10.^(-decades * (0:r-1)'/max(ref-1,1));

cr = cell(d,1);
cr{1} = zeros(1, n, r);
for j = 1:r
    u = randn(n,1);  u = u/norm(u);
    cr{1}(1,:,j) = w(j) * u;
end
for k = 2:d-1
    G = zeros(r, n, r);                     % diagonal in the term index
    for j = 1:r
        u = randn(n,1);  u = u/norm(u);
        G(j,:,j) = u;
    end
    cr{k} = G;
end
cr{d} = zeros(r, n, 1);
for j = 1:r
    u = randn(n,1);  u = u/norm(u);
    cr{d}(j,:,1) = u;
end

v = cell2core(tt_tensor, cr);
v = v / norm(v);
end
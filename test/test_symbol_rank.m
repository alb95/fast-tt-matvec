%TEST_SYMBOL_RANK  How accurate is the truncated resolvent symbol?
%
%   A = (S x..x S) diag(F) (S x..x S)' with F(j1..jd) = 1/(mu_j1+..+mu_jd-sigma),
%   so the TT rank of A is that of the array F, and exp2_resonance truncates F
%   until that rank meets a cap.  This script asks what the cap costs.
%
%   WHY THE FROBENIUS ERROR IS NOT THE ANSWER.  Near the resonance F is
%   dominated by a single huge entry, 1/delta at the resonant multi-index.  A
%   TT-SVD captures that entry with one rank-one term, so the RELATIVE
%   FROBENIUS error collapses almost immediately and says nothing about the
%   remaining entries -- which are exactly the ones that carry the
%   non-resonant part of x = A v.  Three measures are therefore reported.
%
%     frob      ||F - F_r||_F / ||F||_F               flattered by the spike
%     on A v    ||(F - F_r) .* vhat|| / ||F .* vhat|| what the experiment sees
%     entrywise max_j |F_j - F_r(j)| / |F_j|          the harsh one
%
%   plus the error committed on the resonant coefficient alone, which is what
%   the answer is made of, and a table of the rank each tolerance demands.
%
%   Same parameters, same seed and same v as exp2_resonance, so the numbers
%   apply to the run reported in the paper.

clear; clc; format compact; rng(4)

d      = 3;
n      = 512;
JABC   = [5 9 13];
RV     = 50;
DTIME  = 1e-6;                  % the detuning the timings are taken at
RANKS  = [2 4 6 8 10 12 16 20 30];
TOLS   = [1e-4 1e-6 1e-8 1e-10 1e-12];

fprintf('\n  TEST_SYMBOL_RANK   d=%d  n=%d  mode %s  delta/gap=%.0e\n\n', ...
    d, n, mat2str(JABC), DTIME);

%% ---- the symbol, exactly as exp2_resonance builds it ------------------------
h   = 1/(n+1);
lam = (2 - 2*cos(pi*(1:n)'/(n+1)))/h^2;
S   = sqrt(2*h)*sin(pi*(1:n)'*(1:n)/(n+1));
S   = S/norm(S(:,1));

lstar = sum(lam(JABC));
Lsum  = lam;
for k = 2:d, Lsum = reshape(Lsum, [], 1) + reshape(lam, 1, []); Lsum = Lsum(:); end
dist  = abs(Lsum - lstar);
eta   = min(dist(dist > 1e-9*lstar));
sigma = lstar - DTIME*eta;

Fsym = 1./(reshape(Lsum, n*ones(1,d)) - sigma);
clear Lsum dist
fprintf('  lambda* = %.6g, gap = %.4g, delta = %.3e, 1/delta = %.2e\n', ...
    lstar, eta, lstar-sigma, 1/(lstar-sigma));
fprintf('  the resonant entry carries %.2f%% of ||F||_F^2\n\n', ...
    100*Fsym(JABC(1),JABC(2),JABC(3))^2/sum(Fsym(:).^2));

%% ---- the same right-hand side as the experiment ---------------------------
v  = tt_rand(n*ones(d,1), d, RV);  v = v/norm(v);
vh = kron_apply(reshape(full(v), n*ones(1,d)), S.');   % v in the eigenbasis
clear v
xh  = Fsym .* vh;                       % x = A v, in the eigenbasis
nxh = norm(xh(:));
xres = xh(JABC(1), JABC(2), JABC(3));   % the resonant coefficient of x

%% ---- what each tolerance costs in rank ------------------------------------
F0 = tt_tensor(Fsym);
fprintf('  rank demanded by each tolerance\n');
fprintf('    %-12s', 'tolerance');  fprintf('%10.0e', TOLS);  fprintf('\n');
fprintf('    %-12s', 'r^A');
for t = TOLS, fprintf('%10d', max(round(F0, t).r)); end
fprintf('\n\n');

%% ---- what each rank costs in accuracy -------------------------------------
nF = norm(Fsym(:));
fprintf('  %6s | %10s %12s %12s | %12s\n', ...
    'r^A', 'frob', 'on A v', 'entrywise', 'resonant');
fprintf('  %s\n', repmat('-',1,60));
for r = RANKS
    Fr = reshape(full(round(F0, 1e-15, r)), n*ones(1,d));
    E  = Fsym - Fr;
    e_frob = norm(E(:))/nF;
    e_av   = norm(reshape(E.*vh, [], 1))/nxh;
    e_ent  = max(abs(E(:))./abs(Fsym(:)));
    e_res  = abs(E(JABC(1),JABC(2),JABC(3))*vh(JABC(1),JABC(2),JABC(3)))/abs(xres);
    fprintf('  %6d | %10.2e %12.2e %12.2e | %12.2e\n', r, e_frob, e_av, e_ent, e_res);
    clear Fr E
end
fprintf('\n  "on A v" is the one to quote: it is the error the operator truncation\n');
fprintf('  contributes to x = A v, for the v actually used in the experiment.\n\n');

%% ==========================================================================
function T = kron_apply(T, M)
d = ndims(T);  n = size(T,1);
for k = 1:d
    T = reshape(T, n, []);
    T = M*T;
    T = reshape(T, [n, n*ones(1,d-1)]);
    T = permute(T, [2:d 1]);
end
end
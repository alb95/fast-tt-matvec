%TEST_ORDERINGS_AGREE  The orderings within each family return the same object.
%
%   The three STTA orderings -- sequential, fused, pre-sketched -- contract the
%   same network in a different order, so in exact arithmetic they return the
%   same Omega_k, Psi_k and hence the same approximant.  The same holds for the
%   two SRC orderings.  This is the claim made in the paper where the three
%   routes are introduced, and it is what lets each family be drawn as a single
%   accuracy curve in the figures.  This script checks it at a size that runs
%   in a few seconds.
%
%   What is NOT checked here is accuracy against A*v: the approximant is a
%   randomized one and its error is the subject of the experiments, not of a
%   test.  Only agreement between orderings is at stake.
%
%   Run setup first.

clear; clc; format compact; rng(0)

d  = 4;      % order
mn = 64;     % both mode sizes, so the test does not depend on which of the
             % two the helpers read
rA = 3;      % TT rank of the operator
rv = 12;     % TT rank of the argument
s  = 5;      % target rank
p  = 4;      % oversampling

TOL = 1e-10; % the orderings differ only by floating-point associativity

A = rand_tt_operator(d, mn, mn, rA);
v = rand_tt_vector_decay(d, mn, rv, 8, min(6, rv));

Yall = cell(1,d);  for k = 1:d, Yall{k} = randn(mn, s+p); end
Y    = Yall;                                        % STTA uses all s+ell columns
Ysrc = cell(1,d);  for k = 1:d, Ysrc{k} = Yall{k}(:, 1:s); end   % SRC the first s

fprintf('\n  d = %d, m = n = %d, r^A = %d, r^v = %d, s = %d, ell = %d\n\n', ...
        d, mn, rA, rv, s, p);

%% ---- STTA: four routes, one object --------------------------------------
w = cell(1,4);
lab = {'sequential', 'fused', 'pre-sketched', 'pre-sketched (operator reused)'};

w{1} = stta_matvec_sequential(A, v, s, p, Y);
w{2} = stta_matvec_fused(A, v, s, p, Y);
w{3} = stta_matvec_presketched(A, s, p, Y, v);
AY   = stta_sketch_operator(A, s, p, Y);
w{4} = stta_matvec_presketched_reused(A, AY, v, p);

nw   = norm(w{1});
fail = false;
fprintf('  STTA, relative difference from the sequential ordering\n');
for q = 2:4
    e = norm(w{q} - w{1}) / nw;
    fprintf('    %-32s %9.2e%s\n', lab{q}, e, verdict(e, TOL));
    fail = fail || ~(e < TOL);
end

%% ---- SRC: two routes, one object ----------------------------------------
u1 = src_matvec_standard   (A, v, s, Ysrc);
u2 = src_matvec_presketched(A, v, s, Ysrc);
e  = norm(u2 - u1) / norm(u1);
fprintf('\n  SRC, relative difference from the standard ordering\n');
fprintf('    %-32s %9.2e%s\n', 'pre-sketched', e, verdict(e, TOL));
fail = fail || ~(e < TOL);

%% ---- the two families are NOT the same ----------------------------------
%  A reassurance that the test above is not passing vacuously: STTA and SRC
%  build different projectors, so their approximants must differ.
fprintf('\n  STTA against SRC (expected to differ) %11.2e\n', ...
        norm(u1 - w{1}) / nw);

if fail
    fprintf('\n  FAILED: some ordering does not agree to %.0e.\n\n', TOL);
else
    fprintf('\n  passed: every ordering agrees to %.0e.\n\n', TOL);
end

function t = verdict(e, tol)
if e < tol, t = '   ok'; else, t = '   FAILED'; end
end

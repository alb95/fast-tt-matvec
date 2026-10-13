%EXPLORE_ORDERINGS  Quick timing of the three STTA orderings, one run each.
%
%   Pre-sketched, fused and sequential on the same operator, argument and
%   random matrices.  Every route builds everything it needs inside the
%   timed call (the pre-sketched one runs through stta_matvec_presketched, so the
%   pre-sketch is charged).  Each route is run ONCE per configuration: this
%   is for exploring parameters fast, not for the paper.  Single runs are
%   noisy -- differences under about 20 per cent mean nothing.
%
%   Edit the table CFG below: one row per configuration,
%
%       [ d   m   n   r^A   r^v   s   ell ]
%
%   m and n are the two mode sizes of the operator as passed to
%   rand_tt_operator; v and the random matrices take their sizes from the
%   operator itself, so any choice is consistent.
%
%   Options:
%       WARMUP    run every route once on a tiny instance first, so that the
%                 first configuration does not pay MATLAB's first-call costs
%       ERRORS    also compute the relative error of each route against A*v
%                 (forms A*v exactly: expensive at large ranks)
%       DECADES   decay of the argument, as in rand_tt_vector_decay

clear; clc; format compact; rng(0)

%        d     m     n   r^A   r^v    s   ell
CFG = [  10   1000   5    10    30    30    10];

WARMUP  = true;
ERRORS  = false;
DECADES = 8;   REF = 16;

LAB = {'pre-sketched', 'fused', 'sequential'};

%% ---- warm-up ------------------------------------------------------------
if WARMUP
    A = rand_tt_operator(3, 16, 16, 2);
    v = rand_tt_vector_decay(3, A.m(1), 4, DECADES, 4);
    Y = cell(1,3);  for k = 1:3, Y{k} = randn(A.n(k), 6); end
    stta_matvec_presketched(A, 4, 2, Y, v);
    stta_matvec_fused(A, v, 4, 2, Y);
    stta_matvec_sequential(A, v, 4, 2, Y);
end

%% ---- the runs -----------------------------------------------------------
fprintf('\n  %2s %5s %5s %4s %4s %4s %4s | %10s %10s %10s | %7s %7s', ...
    'd', 'm', 'n', 'r^A', 'r^v', 's', 'ell', ...
    'pre (s)', 'fused (s)', 'seq (s)', 'fus/pre', 'seq/pre');
if exist('ops_model', 'file') == 2, fprintf(' | %9s', 'ops f/p'); end
if ERRORS, fprintf(' | %9s %9s %9s', 'e pre', 'e fused', 'e seq'); end
fprintf('\n  %s\n', repmat('-', 1, 110));

T = nan(size(CFG,1), 3);
for c = 1:size(CFG,1)
    d = CFG(c,1);  m = CFG(c,2);  n = CFG(c,3);
    rA = CFG(c,4);  rv = CFG(c,5);  s = CFG(c,6);  p = CFG(c,7);  l = s + p;

    A = rand_tt_operator(d, m, n, rA);
    v = rand_tt_vector_decay(d, A.m(1), rv, DECADES, min(REF, rv));
    Y = cell(1,d);  for k = 1:d, Y{k} = randn(A.n(k), l); end

    f = { @() stta_matvec_presketched(A, s, p, Y, v)
        @() stta_matvec_fused(A, v, s, p, Y)
        @() stta_matvec_sequential(A, v, s, p, Y) };

    w = cell(1,3);
    for q = 1:3
        try
            tic;  w{q} = f{q}();  T(c,q) = toc;
        catch ME
            fprintf('  [%s failed: %s]\n', LAB{q}, ME.message);
        end
    end

    fprintf('  %2d %5d %5d %4d %4d %4d %4d | %10.4f %10.4f %10.4f | %6.2fx %6.2fx', ...
        d, m, n, rA, rv, s, p, T(c,:), T(c,2)/T(c,1), T(c,3)/T(c,1));
    if exist('ops_model', 'file') == 2
        C = ops_model(d, A.m(1), A.n(1), rA, rv, s, p);   % m = columns (mode of v), n = rows
        fprintf(' | %8.2fx', C.fused/C.ours);
    end
    if ERRORS
        ex = A*v;  nex = norm(ex);
        e = nan(1,3);
        for q = 1:3, if ~isempty(w{q}), e(q) = norm(w{q} - ex)/nex; end, end
        fprintf(' | %9.1e %9.1e %9.1e', e);
    end
    fprintf('\n');
    clear A v Y w f
end
fprintf('\n  one run per route: ratios within ~20%% of each other are noise.\n\n');
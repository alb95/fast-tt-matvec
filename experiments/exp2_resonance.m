%EXP2_RESONANCE  Resonant linear response: A b compressed far below the rank of b.
%
%   THE SETTING, AND WHY IT IS NOT AN ITERATION.  Given a discretised operator
%   L on a tensor grid and a FIXED excitation b, the response at a detuning
%   sigma is
%
%       x(sigma) = (L - sigma I)^{-1} b,
%
%   and one evaluates it for a whole sweep of sigma.  This is the frequency
%   response of a structure or a cavity, the resolvent analysis of a flow, the
%   Green function (E + i eta - H)^{-1} v swept over E, and the shift solves of
%   interpolatory model reduction.  Nothing here is iterative: the solves are
%   independent of one another, b is given data and is never updated, and
%   sigma is prescribed by the application rather than by a convergence
%   history.  There is no "after the first step the argument is aligned".
%
%   WHY THE ANSWER IS OF LOW RANK NEAR A RESONANCE.  In the eigenbasis of L,
%
%       x(sigma) = sum_j  b_j / (lambda_j - sigma)  phi_j .
%
%   As sigma approaches an eigenvalue lambda*, the terms of that eigenspace
%   are amplified by 1/delta, delta = |sigma - lambda*|, and every other term
%   by at most 1/gap.  The response therefore collapses onto the resonant
%   eigenspace, and for a Kronecker sum that eigenspace is spanned by
%   SEPARABLE vectors: it has low TT rank.  The excitation b keeps its own
%   rank r^v throughout.  High rank in, low rank out, with both ends given by
%   the problem.
%
%   THE LAW THE SWEEP EXHIBITS.  The relative error of the rank-s truncation,
%   once s reaches the TT rank of the eigenspace, is proportional to the
%   detuning:  ||x - trunc_s(x)||/||x|| ~ (delta/gap) ||b_perp||/|b_*| .  The
%   table below shows it decade by decade, which is what makes the example
%   verifiable rather than anecdotal.
%
%   THE RESONANT EIGENSPACE.  lambda* = mu_a + mu_b + mu_c with a, b, c
%   DISTINCT has multiplicity d! = 6, and its eigenspace is spanned by the
%   permutations of phi_a (x) phi_b (x) phi_c: six orthonormal separable
%   vectors, so its TT rank is exactly d = 3.  Below s = 3 the answer is not
%   representable at all, whatever the detuning; at and above s = 3 the error
%   is the detuning error.  Both facts are visible in the table.
%
%   A NOTE ON DAMPING.  A physically damped response has sigma complex, the
%   damping playing the role of delta.  The rank mechanism is identical; we
%   keep sigma real so that every quantity, sketches included, stays real.
%
%   THE ROUTES, all computing a rank-s approximation of x = A b.
%       form + round          A*b at TT-rank r^A r^v, then TT-rounded
%       STTA, sequential      the cores of A*b formed one at a time
%       STTA, fused           both sweeps on the raw cores, environment reused
%       STTA, pre-sketched    stta_matvec_presketched, the pre-sketch INSIDE the timer
%       SRC, standard         both sweeps on the raw cores (..._raw)
%       SRC, pre-sketched     pre-sketched first sweep, absorbed second sweep
%
%   Per interior core, in units of (r^A)^2 m n, the weights are 2(r^v)^2 for
%   the sequential, r^v(2s+ell) for the fused, (s+ell)(1+s) for the
%   pre-sketched, 2 r^v s for SRC and s + 2s^2 for SRC absorbed.  The factor
%   (r^A)^2 m n is common to all of them, so r^A and n set the absolute cost
%   but not the ordering: only r^v/s does.  A sweep evaluates the response at
%   many sigma, so whatever one solve saves is multiplied by the sweep.
%
%   OUTPUT
%       response_acc.csv    accuracy: one row per s, one block per detuning
%       response_time.csv   one row per s: errors and seconds of every route
%       response.txt        parameters, ranks, object sizes

clear; clc; format compact; rng(4)
line = @() fprintf('%s\n', repmat('-',1,116));

%% ---- parameters -----------------------------------------------------------
d      = 3;
n      = 512;            % mode size: large, so that s + ell << n
JABC   = [5 9 13];       % the targeted mode: distinct indices, multiplicity d!
RV     = 100;            % TT-rank of the excitation
DFRAC  = [1e-1 1e-3 1e-5 1e-6];   % detuning delta, as a fraction of the local gap
DTIME  = 1e-6;                    % the detuning used for the timings
RA_MAX = 16;             % cap on r^A: TOL_A is relaxed until the symbol meets it
TOL_A0 = 1e-10;
ell    = 4;              % oversampling, as in every experiment
SVALS  = [3 5 10 20 30];
FORM_MAXGB = 3;          % skip form+round above this core size
SEQ_MAXGB  = 3;          % skip the sequential route above this core size
REPS   = 5;      % passes of the round-robin timer; the minimum is reported
TARGET = 0.25;

LAB  = {'form + round','STTA sequential','STTA fused','STTA pre-sketched', ...
        'SRC standard','SRC pre-sketched'};
IPRE = 4;

assert(max(SVALS) + ell <= n, 's + ell must not exceed the mode size n.');
assert(numel(unique(JABC)) == d, 'JABC must have distinct entries, for multiplicity d!.');

fprintf('\n'); line();
fprintf('  EXP2_RESONANCE   d=%d  n=%d  r^v=%d  ell=%d  s = %s\n', d, n, RV, ell, mat2str(SVALS));
line();

%% ---- the operator L and the target eigenpair -------------------------------
h   = 1/(n+1);
lam = (2 - 2*cos(pi*(1:n)'/(n+1)))/h^2;
S   = sqrt(2*h)*sin(pi*(1:n)'*(1:n)/(n+1));
S   = S/norm(S(:,1));

lstar = sum(lam(JABC));
Lsum  = lam;
for k = 2:d, Lsum = reshape(Lsum, [], 1) + reshape(lam, 1, []); Lsum = Lsum(:); end
dist  = abs(Lsum - lstar);
mult  = sum(dist < 1e-9*lstar);
eta   = min(dist(dist > 1e-9*lstar));
fprintf('  lambda* = %.6g  (mode %s)  multiplicity %d = d!  next eigenvalue at %.4g\n', ...
        lstar, mat2str(JABC), mult, eta);

% the eigenspace: the d! permutations of phi_a (x) phi_b (x) phi_c, each a
% rank-one TT tensor and mutually orthonormal.  Its TT rank is d.
P   = perms(JABC);
Phi = cell(1, size(P,1));
for q = 1:size(P,1)
    cr = cell(d,1);
    for i = 1:d, cr{i} = reshape(S(:,P(q,i)), [1 n 1]); end
    Phi{q} = cell2core(tt_tensor, cr);
end
fprintf('  eigenspace: %d orthonormal separable vectors, TT rank %d\n', numel(Phi), d);

%% ---- the starting vector ---------------------------------------------------
% The excitation: a TT vector of prescribed rank r^v.  It is the data of the
% problem, fixed once and reused at every sigma of the sweep.
v  = tt_rand(n*ones(d,1), d, RV);  v = v/norm(v);
rv = max(v.r);
vf = reshape(full(v), n*ones(1,d));
vh = kron_apply(vf, S');                       % v in the eigenbasis of L
clear vf
cres = 0;
for q = 1:numel(Phi), cres = cres + dot(Phi{q}, v)^2; end
cres = sqrt(cres);
fprintf('  excitation b: TT rank r^v = %d, overlap with the eigenspace %.3e\n\n', rv, cres);

%% ---- accuracy: trunc_s(A v) against A v, and against the eigenspace --------
EA = nan(numel(SVALS), numel(DFRAC));      % relative error of the rank-s truncation
EG = nan(numel(SVALS), numel(DFRAC));      % sine of the angle to the eigenspace
RAQ = zeros(1, numel(DFRAC));  RKX = zeros(1, numel(DFRAC));
fprintf('  ACCURACY of trunc_s(A b).  Columns: delta/gap, and the angle to the eigenspace.\n');
fprintf('    %-6s', 's');
for iq = 1:numel(DFRAC), fprintf('%12.0e%12s', DFRAC(iq), '(angle)'); end
fprintf('\n');
for iq = 1:numel(DFRAC)
    sigma = lstar - DFRAC(iq)*eta;
    Fsym  = 1./(reshape(Lsum, n*ones(1,d)) - sigma);
    F0 = tt_tensor(Fsym);            % the TT-SVD, once per detuning
    clear Fsym
    TOL_A = TOL_A0;  Ftt = round(F0, TOL_A);
    while max(Ftt.r) > RA_MAX && TOL_A < 1e-2
        TOL_A = TOL_A*10;  Ftt = round(F0, TOL_A);
    end
    clear F0
    RAQ(iq) = max(Ftt.r);
    Ff   = reshape(full(Ftt), n*ones(1,d));
    xf   = kron_apply(Ff .* vh, S);
    xr   = round(tt_tensor(xf), 1e-14);
    nx   = norm(xr);
    RKX(iq) = max(round(xr, 1e-6).r);
    for is = 1:numel(SVALS)
        xs = round(xr, 1e-14, SVALS(is));
        EA(is,iq) = norm(xs - xr)/nx;
        c = 0;
        for q = 1:numel(Phi), c = c + dot(Phi{q}, xs)^2; end
        EG(is,iq) = sqrt(max(0, 1 - c/norm(xs)^2));
    end
    if DFRAC(iq) == DTIME
        Ftime = Ftt;  TOL_AT = TOL_A;  xref = xr;  nxref = nx;
    end
    clear Ff xf xr
end
for is = 1:numel(SVALS)
    fprintf('    %-6d', SVALS(is));
    for iq = 1:numel(DFRAC), fprintf('%12.2e%12.2e', EA(is,iq), EG(is,iq)); end
    fprintf('\n');
end
fprintf('    r^A:  ');
for iq = 1:numel(DFRAC), fprintf('%12d%12s', RAQ(iq), ''); end
fprintf('\n');
fprintf('    rk(x):');
for iq = 1:numel(DFRAC), fprintf('%12d%12s', RKX(iq), ''); end
fprintf('   <- numerical TT rank of x at 1e-6\n\n');
clear vh
write_csv('response_acc.csv', ...
    [{'s'}, arrayfun(@(q) sprintf('e_d%g',q), DFRAC, 'uni', 0), ...
            arrayfun(@(q) sprintf('ang_d%g',q), DFRAC, 'uni', 0)], [SVALS(:) EA EG]);

%% ---- the operator at the timing shift --------------------------------------
A  = round(kron_op(S,d) * diag_tt(Ftime) * kron_op(S',d), 1e-12);
rA = max(A.r);
R  = rA*rv;
coreGB = 8*n*R^2/1e9;
allGB  = 8*n*(2*R + (d-2)*R^2)/1e9;
if TOL_AT > 1e-8
    fprintf(['  NOTE: the cap r^A <= %d forced the operator tolerance to %.0e, so the\n' ...
             '        errors below that level are measured against the TRUNCATED operator.\n'], RA_MAX, TOL_AT);
end
fprintf('  timing at delta/gap = %.0e:  r^A = %d at tolerance %.0e,  r^A r^v = %d\n', DTIME, rA, TOL_AT, R);
fprintf('  A*b: one interior core %.2f GB, all cores %.2f GB\n', coreGB, allGB);
runForm = coreGB <= FORM_MAXGB;
runSeq  = coreGB <= SEQ_MAXGB;
if ~runForm, fprintf('  -> form+round SKIPPED (core exceeds %g GB)\n', FORM_MAXGB); end
if ~runSeq,  fprintf('  -> sequential SKIPPED (core exceeds %g GB)\n', SEQ_MAXGB);  end
fprintf('\n');

%% ---- the comparison --------------------------------------------------------
Yall = cell(1,d);
for k = 1:d, Yall{k} = randn(n, max(SVALS) + ell); end

Hdr = {'s','e_floor','e_form','e_seq','e_fused','e_pre','e_src','e_src_abs','e_trunc', ...
       't_form','t_seq','t_fused','t_pre','t_src','t_src_abs'};
T = nan(numel(SVALS), numel(Hdr));

for ii = 1:numel(SVALS)
    s = SVALS(ii);  l = s + ell;
    Y  = cell(1,d);  for k = 1:d, Y{k}  = Yall{k}(:,1:l); end
    Ys = cell(1,d);  for k = 1:d, Ys{k} = Yall{k}(:,1:s); end

    f = { @() round(A*v, 1e-14, s)
          @() stta_matvec_sequential(A, v, s, ell, Y)
          @() stta_matvec_fused(A, v, s, ell, Y)
          @() stta_matvec_presketched(A, s, ell, Y, v)
          @() src_matvec_standard(A, v, s, Ys)
          @() src_matvec_presketched(A, v, s, Ys) };
    if ~runForm, f{1} = []; end
    if ~runSeq,  f{2} = []; end

    e = nan(1,6);
    for q = 1:6
        if isempty(f{q}), continue; end
        try
            e(q) = norm(f{q}() - xref)/nxref;
        catch ME
            fprintf('  [%s failed at s = %d: %s]\n', LAB{q}, s, ME.message);
            f{q} = [];
        end
    end
    % truncate the excitation first, then apply the resolvent exactly: the
    % cheap alternative, charged nothing for its own approximation
    etrunc = norm(round(A*round(v, 1e-14, s), 1e-13) - xref)/nxref;

    t = time_best_multi(f, REPS, TARGET);
    T(ii,:) = [s EA(ii, DFRAC == DTIME) e etrunc t];

    fprintf('=== s = %d   (r^v/s = %.1f) ===\n', s, rv/s);
    fprintf('    %-20s %10s %9s %8s\n', 'route', 'error', 'seconds', 'x pre');
    fprintf('    %-20s %10.2e\n', 'best rank-s', EA(ii, DFRAC == DTIME));
    for q = 1:6
        if isnan(t(q))
            fprintf('    %-20s %10s %9s\n', LAB{q}, '--', 'not run');
        else
            fprintf('    %-20s %10.2e %9.4f %7.2fx\n', LAB{q}, e(q), t(q), t(q)/t(IPRE));
        end
    end
    fprintf('    %-20s %10.2e\n', 'truncate b first', etrunc);
    for q = [2 3]
        if ~isnan(e(q)) && abs(e(q)-e(IPRE)) > 1e-6*max(e(IPRE),1e-12) + 1e-12
            fprintf('    CHECK: %s and pre-sketched differ (%.2e vs %.2e)\n', LAB{q}, e(q), e(IPRE));
        end
    end
    if abs(e(5)-e(6)) > 1e-6*max(e(5),1e-12) + 1e-12
        fprintf('    CHECK: the two SRC differ (%.2e vs %.2e)\n', e(5), e(6));
    end
    fprintf('\n');
end
line();
write_csv('response_time.csv', Hdr, T);

fprintf('\n  TIMES relative to STTA pre-sketched\n');
fprintf('    %-20s', 's');  fprintf('%9d', SVALS);  fprintf('\n');
for q = [1 2 3 5 6]
    fprintf('    %-20s', LAB{q});  fprintf('%9.2f', T(:,9+q)./T(:,9+IPRE));  fprintf('\n');
end
fprintf('\n  predicted from the weights (flops only, r^A and n cancel)\n');
fprintf('    %-20s', 's');  fprintf('%9d', SVALS);  fprintf('\n');
pre = (SVALS+ell).*(1+SVALS);
fprintf('    %-20s', LAB{2});  fprintf('%9.2f', 2*rv^2./pre);          fprintf('\n');
fprintf('    %-20s', LAB{3});  fprintf('%9.2f', rv*(2*SVALS+ell)./pre); fprintf('\n');
fprintf('    %-20s', LAB{5});  fprintf('%9.2f', 2*rv*SVALS./pre);       fprintf('\n');
fprintf('    %-20s', LAB{6});  fprintf('%9.2f', (SVALS+2*SVALS.^2)./pre);fprintf('\n\n');

%% ---- parameters and sizes ---------------------------------------------------
fid = fopen('response.txt', 'w');
fprintf(fid, 'exp2_resonance   resonant linear response, one point of a sweep\n');
fprintf(fid, 'd %d\nn %d\nell %d\nsvals %s\ndfrac %s\ndtime %g\n', d, n, ell, ...
        mat2str(SVALS), mat2str(DFRAC), DTIME);
fprintf(fid, 'mode %s\nlambda_star %.10g\nmultiplicity %d\nspectral_gap %.6g\n', ...
        mat2str(JABC), lstar, mult, eta);
fprintf(fid, 'rv %d\noverlap_with_eigenspace %.4g\n', rv, cres);
for iq = 1:numel(DFRAC)
    fprintf(fid, 'rA at delta/gap = %-8g %d   rank_x_1e-6 %d\n', DFRAC(iq), RAQ(iq), RKX(iq));
end
fprintf(fid, 'rA_timing %d\ntolA_timing %g\nrArv %d\n', rA, TOL_AT, R);
fprintf(fid, 'reps %d\ntarget %g\n', REPS, TARGET);
fprintf(fid, 'matlab %s\nthreads %d\ncomputer %s\ndate %s\n', ...
        version, maxNumCompThreads, computer, datestr(now, 31));
fprintf(fid, '\nobject sizes, megabytes, double precision\n');
fprintf(fid, 'resolvent A         %.1f\n', 8*numel(A.core)/1e6);
fprintf(fid, 'excitation b        %.1f\n', 8*numel(v.core)/1e6);
fprintf(fid, 'one core of A*b     %.1f\n', 1e3*coreGB);
fprintf(fid, 'all cores of A*b    %.1f\n', 1e3*allGB);
for s = SVALS
    fprintf(fid, 'A sketched, s = %-3d %.1f\n', s, 8*(2*rA + (d-2)*rA^2)*n*(s+ell)/1e6);
end
fprintf(fid, 'form_round_run %d\nsequential_run %d\n', runForm, runSeq);
fclose(fid);
fprintf('  written response.txt\n\n');

%% ==========================================================================
function M = kron_op(S, d)
    n = size(S,1);  c = cell(d,1);
    for k = 1:d, c{k} = reshape(S, [1 n n 1]); end
    M = cell2core(tt_matrix, c);
end

function D = diag_tt(F)
    cr = core2cell(F);  d = numel(cr);  c = cell(d,1);
    for k = 1:d
        [r1, nk, r2] = size(cr{k});
        Gk = zeros(r1, nk, nk, r2);
        for i = 1:nk, Gk(:,i,i,:) = reshape(cr{k}(:,i,:), [r1 1 1 r2]); end
        c{k} = Gk;
    end
    D = cell2core(tt_matrix, c);
end

function T = kron_apply(T, M)
    d = ndims(T);  n = size(T,1);
    for k = 1:d
        T = reshape(T, n, []);
        T = M*T;
        T = reshape(T, [n, n*ones(1,d-1)]);
        T = permute(T, [2:d 1]);
    end
end
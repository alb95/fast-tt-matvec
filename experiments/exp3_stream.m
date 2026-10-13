%EXP3_STREAM  A sum over a stream of realisations, compressed once.
%
%   N realisations v_i = m + rho xi_i of a random field arrive one at a time.
%   The coefficients of the fluctuations have zero column sums, so
%
%       sum_i A v_i = N A m
%
%   exactly: the fluctuations cancel, and the answer is smaller than its terms
%   by the ratio rho.  Rounding a single term to rank s therefore commits an
%   error relative to THAT TERM, which is large relative to the sum.
%
%   THREE ROUTES.
%
%     ours        the sketches Omega_k, Psi_k are linear in the argument, so
%                 each realisation adds its contribution to two accumulators
%                 and is then discarded.  ONE recovery at the end, nothing
%                 truncated before it, and the memory does not depend on N.
%     SRC         its second sweep projects with an orthonormal factor computed
%                 from the argument, so the compressions of two realisations
%                 cannot be added.  What CAN be done -- and is done here -- is
%                 to keep the first sweep, which is linear, sum it, and
%                 orthogonalise the sum once per site
%                 (src_sketch + src_recover_sum).  The
%                 price is that site k needs Q_{k+1} and every realisation
%                 again: d-1 passes over the stream, or the stream in memory.
%     trunc-1st   round v_i to rank s, apply A exactly, add, round again.  The
%                 route a user takes without any of this machinery.
%
%   WHAT IS TIMED.  The realisations are generated ONCE, outside the timed
%   region, and all three routes read the same array: the generator is not part
%   of any method.  Each route builds what depends on the operator -- the
%   prepared operator for ours, the sketched operator for SRC -- INSIDE its own
%   timed call, once for the whole stream, which is exactly the amortisation
%   the Krylov section claims.  The three routes are timed round-robin and
%   repeated (time_best_multi), and the run also times the operator-dependent
%   part alone, so that the table can report what fraction of the work it is.
%
%   THE MEMORY COLUMN is what each route must hold while the stream runs: two
%   accumulators for ours, the first-sweep sketches AND the realisations for
%   SRC, one running TT tensor for trunc-1st.
%
%   TWO SCENARIOS, the same sweep in s run twice.
%     exact      the column sums of the coefficients vanish, so sum_i A v_i is
%                exactly N A m, of TT rank r^A r^u.
%     residual   a fluctuation of relative size EPSR survives the sum, so the
%                exact answer is N A m + rho A Xi and is NO LONGER of low rank.
%                The best rank-s error then plateaus at EPSR, and the question
%                is which routes see that plateau and which do not.
%   In both the reference is the exact sum, so the comparison is the same one.
%
%   PART D sweeps EPS at a fixed s and is reported as a table: the cost does
%   not depend on EPS, so there is nothing to time there.
%
%   OUTPUT
%       stream.csv           exact cancellation: one row per s
%       stream_eps.csv       residual EPSR:      one row per s
%       stream_residual.csv  part D, the sweep in EPS
%       stream.txt           parameters, sizes, amortisation

clear; clc; format compact; rng(11)
line = @() fprintf('%s\n', repmat('-',1,120));

d     = 4;
n     = 300;
rA    = 4;         % TT rank of the operator, dense cores
Q     = 50;        % rough modes in each fluctuation  ->  r^v = J + Q
J     = 6;         % modes of the mean field, with a decaying spectrum
BET   = 10.^-(0:5);% weights: one decade each
N     = 100;       % realisations in the stream
RHO   = 20;        % fluctuation / signal within a single realisation
SVALS = [4 8 12 16 20 24];
PVALS = [2 4 6 8 10 12];
REPS  = 5;         % passes of the round-robin timer over the three routes
EPSR  = 1e-3;      % residual left in the sum by the second scenario

h  = 1/(n+1);
xg = (1:n)'*h;

fprintf('\n'); line();
fprintf('  EXP3_STREAM   d=%d  n=%d  r^A=%d  Q=%d  N=%d  rho=%d\n', d, n, rA, Q, N, RHO);
line();

%% ---- the operator: dense cores, moderate TT rank -------------------------
A  = rand_ttm(d, n, rA);
rA = max(A.r);

%% ---- the mean field and the modes of the fluctuation ---------------------
bet = BET;
m = [];
for j = 1:J
    cj = cell(d,1);
    c0 = 0.25 + 0.5*rand(1,d);  w0 = 0.10 + 0.10*rand;
    for i = 1:d, cj{i} = reshape(exp(-((xg-c0(i)).^2)/w0^2), [1 n 1]); end
    mj = bet(j)*cell2core(tt_tensor, cj);
    if isempty(m), m = mj; else, m = m + mj; end
end
m = round(m, 1e-14)/norm(m);

B = cell(1,d);
for i = 1:d
    Bi = zeros(n,Q);
    for q = 1:Q
        Bi(:,q) = cos(pi*(6+2*q)*xg + 2*pi*rand);
        Bi(:,q) = Bi(:,q)/norm(Bi(:,q));
    end
    B{i} = Bi;
end

% Coefficients with zero column sums
Cc = randn(N,Q);  Cc = Cc - mean(Cc,1);
Cc = Cc / mean(sqrt(sum(Cc.^2,2)));

crMid = cell(1,d);
for i = 2:d-1
    Gi = zeros(Q,n,Q);
    for q = 1:Q, Gi(q,:,q) = B{i}(:,q).'; end
    crMid{i} = Gi;
end
crLast = zeros(Q,n,1);
for q = 1:Q, crLast(q,:,1) = B{d}(:,q).'; end
% The stream is generated outside every timed call: the generator belongs to
% the problem, not to a method.  make_stream returns the realisations and the
% EXACT sum of the products, which is the reference in both scenarios.
[Vs, xref] = make_stream(A, m, B, crMid, crLast, Cc, RHO, n, d, Q);
rv = max(Vs{1}.r);  nx = norm(xref);
fprintf('  signal / fluctuation within one realisation: %.3f\n', norm(m)/norm(Vs{1}-m));
fprintf('  each realisation: r^v = %d        the answer: rank %d\n', rv, max(xref.r));
fprintf('  the exact sum of the products carries rank N r^A r^v = %d\n', N*rA*rv);
fprintf('  storing the stream: %.0f MB\n\n', N*8*numel(Vs{1}.core)/1e6);

%% ==========================================================================
%% The sweep in s, run once with exact cancellation and once with a residual
%% ==========================================================================
H = {'s', 'ell', 'floor', 'e_ours', 'e_src', 'e_trunc', ...
     't_ours', 't_src', 't_trunc', 'mem_ours', 'mem_src', ...
     't_prep_ours', 't_prep_src'};

fprintf('  SCENARIO 1 -- exact cancellation, %d realisations\n', N);
C1 = sweep_s(A, Vs, xref, SVALS, PVALS, n, d, N, REPS);
write_csv('stream.csv', H, C1);
clear Vs

% the second scenario: a fluctuation of relative size EPSR survives the sum,
% so the exact answer is no longer of low rank
del = randn(1,Q);
Xi0 = fluct(B, crMid, crLast, del, n, d, Q);
sc  = EPSR*norm(round(N*(A*m),1e-14))/(RHO*norm(A*Xi0));
Ce  = Cc + (sc*del)/N;
[Ve, xrefe] = make_stream(A, m, B, crMid, crLast, Ce, RHO, n, d, Q);
fprintf('\n  SCENARIO 2 -- residual of relative size %.0e, answer of rank %d\n', ...
        EPSR, max(round(xrefe,1e-8).r));
C2 = sweep_s(A, Ve, xrefe, SVALS, PVALS, n, d, N, REPS);
write_csv('stream_eps.csv', H, C2);
clear Ve

fprintf('\n  MEMORY   ours %.1f MB against SRC %.1f MB\n', ...
        max(C1(:,10))/1e6, max(C1(:,11))/1e6);
fprintf('  AMORTISATION   the operator-dependent part is %.1f%% to %.1f%% of our total\n', ...
        100*min(C1(:,12)./C1(:,7)), 100*max(C1(:,12)./C1(:,7)));

%% ==========================================================================
%% PART D.  What if the cancellation is not exact?
%% ==========================================================================
SD = 12; PD = 6; lD = SD + PD;
fprintf('\n  PART D -- imperfect cancellation, at s = %d\n', SD);
fprintf('  a residual of relative size EPS survives the sum, so EPS is the floor\n');
fprintf('  %10s | %11s %12s %12s %14s | %s\n', 'EPS', 'floor', 'err ours', 'err SRC', 'err trunc-1st', 'verdict');
line();

EPSV = [1e-1 1e-2 1e-3 0];
YD     = cell(1,d);  for i = 1:d, YD{i}     = randn(n, lD); end
YD_src = cell(1,d);  for i = 1:d, YD_src{i} = YD{i}(:,1:SD); end

xm  = round(N*(A*m), 1e-14);  nxm = norm(xm);
D = nan(numel(EPSV), 5);

for jj = 1:numel(EPSV)
    ep = EPSV(jj);
    Cd = Cc;                                   % the centred coefficients
    if ep > 0
        del = randn(1,Q);
        Xi0 = fluct(B, crMid, crLast, del, n, d, Q);
        sc  = ep*nxm/(RHO*norm(A*Xi0));        % scale the residual to EPS
        Cd  = Cd + (sc*del)/N;                 % column sums become sc*del
    end
    Xi   = fluct(B, crMid, crLast, sum(Cd,1), n, d, Q);
    flo  = RHO*norm(A*Xi)/nxm;                 % the floor this residual imposes

    Vd = make_stream(A, m, B, crMid, crLast, Cd, RHO, n, d, Q);

    ea = norm(run_ours(A, SD, PD, YD, Vd)  - xm)/nxm;
    ec = norm(run_src(A, SD, YD_src, Vd)   - xm)/nxm;
    eb = norm(run_trunc(A, SD, Vd)         - xm)/nxm;

    D(jj,:) = [ep flo ea ec eb];
    vd = 'tracks the floor';
    if ea > 20*max(flo,1e-12), vd = 'LOOK: well above the floor'; end
    fprintf('  %10.0e | %11.2e %12.2e %12.2e %14.2e | %s\n', ep, flo, ea, ec, eb, vd);
    clear Vd
end
write_csv('stream_residual.csv', {'eps','floor','e_ours','e_src','e_trunc'}, D);
line();

%% ---- parameters and sizes --------------------------------------------------
fid = fopen('stream.txt', 'w');
fprintf(fid, 'exp3_stream\n');
fprintf(fid, 'd %d\nn %d\nrA %d\nrv %d\nN %d\nrho %d\nQ %d\nJ %d\n', d, n, rA, rv, N, RHO, Q, J);
fprintf(fid, 'svals %s\npvals %s\nreps %d\n', mat2str(SVALS), mat2str(PVALS), REPS);
fprintf(fid, 'matlab %s\nthreads %d\ncomputer %s\ndate %s\n', ...
        version, maxNumCompThreads, computer, datestr(now, 31));
fprintf(fid, '\nobject sizes, megabytes\n');
fprintf(fid, 'operator A              %.1f\n', 8*numel(A.core)/1e6);
fprintf(fid, 'one realisation         %.2f\n', 8*numel(Vs{1}.core)/1e6);
fprintf(fid, 'the whole stream        %.1f\n', N*8*numel(Vs{1}.core)/1e6);
fprintf(fid, 'exact sum, TT rank      %d\n', N*rA*rv);
fclose(fid);
fprintf('\n  written stream.csv, stream_residual.csv, stream.txt\n\n');

%% ==========================================================================
function [Vs, xref] = make_stream(A, m, B, crMid, crLast, Cf, RHO, n, d, Q)
%MAKE_STREAM  The realisations, and the EXACT sum of the products.
%   sum_i A v_i = N A m + rho A Xi with Xi the fluctuation carrying the column
%   sums of Cf; when those vanish the second term is zero.
    N  = size(Cf,1);
    Vs = cell(1,N);
    for i = 1:N, Vs{i} = m + RHO*fluct(B, crMid, crLast, Cf(i,:), n, d, Q); end
    Xi   = fluct(B, crMid, crLast, sum(Cf,1), n, d, Q);
    xref = round(N*(A*m) + RHO*(A*Xi), 1e-14);
end

function C = sweep_s(A, Vs, xref, SVALS, PVALS, n, d, N, REPS)
%SWEEP_S  Accuracy, time and retained memory of the three routes, per s.
    nx = norm(xref);
    line = @() fprintf('%s\n', repmat('-',1,120));
    fprintf('  %4s | %9s %10s %10s %12s | %9s %9s %11s | %9s %9s\n', 's', ...
            'floor', 'err ours', 'err SRC', 'err trunc-1st', 't ours', 't SRC', ...
            't trunc-1st', 'mem ours', 'mem SRC');
    line();
    C = nan(numel(SVALS), 13);
    for ii = 1:numel(SVALS)
        s = SVALS(ii);  p = PVALS(ii);  l = s + p;
        Yk   = cell(1,d);  for i = 1:d, Yk{i}   = randn(n, l); end
        Ysrc = cell(1,d);  for i = 1:d, Ysrc{i} = Yk{i}(:,1:s); end

        f = { @() run_ours(A, s, p, Yk, Vs)
              @() run_src(A, s, Ysrc, Vs)
              @() run_trunc(A, s, Vs) };

        e  = [norm(f{1}() - xref)/nx
              norm(f{2}() - xref)/nx
              norm(f{3}() - xref)/nx];
        fl = norm(round(xref, 1e-15, s) - xref)/nx;
        [t, sp] = time_best_multi(f, REPS, 0.25);
        tprep = [time_best(@() stta_prepare_presketched(A, s, p, Yk), 2, 0.1)
                 time_best(@() stta_sketch_operator(A, s, 0, Ysrc), 2, 0.1)];
        [~, memO] = run_ours(A, s, p, Yk, Vs);
        [~, memS] = run_src(A, s, Ysrc, Vs);
        memS = memS + N*8*numel(Vs{1}.core);    % SRC needs the realisations again

        C(ii,:) = [s l fl e.' t memO memS tprep.'];
        fprintf(['  %4d | %9.2e %10.2e %10.2e %12.2e | %9.2f %9.2f %11.2f' ...
                 ' | %6.1f MB %6.1f MB | spread %4.2f\n'], ...
                s, fl, e(1), e(2), e(3), t(1), t(2), t(3), ...
                memO/1e6, memS/1e6, max(sp));
    end
    line();
end

function [w, bytes] = run_ours(A, s, p, Y, Vs)
%RUN_OURS  One pass, two accumulators, one recovery.
%   The prepared operator is built once for the whole stream, inside the timed
%   region; each realisation contributes its sketches and is then not needed.
    P  = stta_prepare_presketched(A, s, p, Y);
    Om = [];  Ps = [];
    for i = 1:numel(Vs)
        [Oi, Pi] = stta_compress_presketched(P, Vs{i});
        if isempty(Om)
            Om = Oi;  Ps = Pi;
        else
            Om = cellfun(@plus, Om, Oi, 'uni', 0);
            Ps = cellfun(@plus, Ps, Pi, 'uni', 0);
        end
    end
    if nargout > 1
        q = whos('Om','Ps');  bytes = sum([q.bytes]);
    end
    w = stta_recover(Om, Ps);
end

function [w, bytes] = run_src(A, s, Ys, Vs)
%RUN_SRC  The first sweep on every realisation, then one orthogonalisation
%   per site for the summed sketches.  The sketched operator is built once for
%   the whole stream, as ours is.
    N = numel(Vs);
    AY = stta_sketch_operator(A, s, 0, Ys);
    Yd_all = cell(1,N);  Lc_all = cell(1,N);
    for i = 1:N
        [Yd_all{i}, Lc_all{i}] = src_sketch(A, Vs{i}, s, Ys, AY);
    end
    if nargout > 1
        q = whos('Yd_all','Lc_all');  bytes = sum([q.bytes]);
    end
    w = src_recover_sum(A, Vs, s, Yd_all, Lc_all);
end

function [w, bytes] = run_trunc(A, s, Vs)
%RUN_TRUNC  Round the argument first, apply the operator exactly, accumulate.
    w = [];
    for i = 1:numel(Vs)
        ti = round(A*round(Vs{i}, 1e-14, s), 1e-14, s);
        if isempty(w), w = ti; else, w = round(w + ti, 1e-14, s); end
    end
    if nargout > 1
        q = whos('w');  bytes = sum([q.bytes]);
    end
end

function w = fluct(B, crMid, crLast, c, n, d, Q)
    cr = cell(d,1);
    cr{1} = reshape(B{1}*diag(c), [1, n, Q]);
    for i = 2:d-1, cr{i} = crMid{i}; end
    cr{d} = crLast;
    w = cell2core(tt_tensor, cr);
end

function A = rand_ttm(d, n, r)
    rr = [1, r*ones(1,d-1), 1];
    c = cell(d,1);
    for k = 1:d
        c{k} = randn(rr(k), n, n, rr(k+1))/sqrt(rr(k)*rr(k+1)*n);
    end
    A = cell2core(tt_matrix, c);
end
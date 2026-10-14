%EXP1_TARGET_RANK  One operator, one argument, three target ranks.
%
%   d = 4, m = n = 512, r^A = 4, r^v = 40, ell = 4.  Everything is fixed --
%   the operator, the argument, and the random matrices of the sketch -- and
%   only the target rank s changes, so the three groups of bars in the figure
%   differ in s and in nothing else.
%
%       s =   6    well below r^v
%       s =  40    equal to r^v
%       s = 100    above r^v, still below the rank r^A r^v = 160 of A*v
%
%   THE TWO SRC VARIANTS are the two ends of the design space:
%
%       SRC           src_matvec_standard: nothing
%                     pre-processed, the first sweep on the raw cores (the
%                     fused order), the shared second sweep;
%       SRC absorbed  src_matvec_presketched: pre-sketched
%                     first sweep, absorbed second sweep, one transposition
%                     of each operator core for both.
%   WHAT THE COUNTS PREDICT, per interior site, in units of r_A^2 n^2:
%
%       ours            s (s + ell)
%       SRC             2 r^v s        (r^v s per sweep)
%       SRC absorbed    s + 2 s^2      (pre-sketch, then the absorbed sweep)
%       forming A*v     (r^v)^2      plus a rounding of weight n (r^A r^v)^3
%
%   so SRC beats the pre-sketched order once s + ell > 2 r^v, the absorbed
%   variant beats SRC while 2 s < 2 r^v roughly, i.e. for s below r^v, and
%   it beats us only if s < ell.  Forming the cores of A*v, whose cost does not
%   depend on s, overtakes everything once s exceeds r^v.
%
%   SAME INSTANCE, SAME RANDOMNESS.  One set of random matrices is drawn,
%   wide enough for the largest s, and each s takes its first s + ell columns
%   (SRC its first s).  The sketches are therefore nested, and a change
%   between two groups of bars cannot be a different draw of the dice.
%
%   OUTPUT
%       exp1_target_rank.csv        one row per s: seconds and errors
%       exp1_target_rank_plot.csv   one row per route, one column per s,
%                                   time divided by that of the pre-sketched
%                                   order -- read directly by the figure
%       exp1_target_rank.txt        parameters, machine, anchor, object sizes
%
%   Runtime: a few minutes, most of it at s = 100.

clear; clc; format compact; rng(0)

%% ---- parameters ---------------------------------------------------------
d  = 4;
m  = 512;  n = m;
rA = 4;
rv = 40;
p  = 4;                                   % ell
SVALS = [6 40 100];

DECADES = 8;          % v's weights fall by 10^-DECADES ...
REF     = 16;         % ... by its REF-th term.  Fixed, NOT tied to s: the
                      % argument must be the same for all groups.

REPS   = 5;      % passes of the round-robin timer; the minimum is reported
TARGET = 0.25;
WARMUP = 90;     % seconds of dense load before any timing; 0 to skip

LAB = {'A*v + round', 'sequential', 'fused', 'pre-sketched', 'SRC', 'SRC absorbed'};
TAG = {'formround', 'sequential', 'fused', 'presketched', 'src', 'srcabsorbed'};

R = rA*rv;
fprintf('\n  EXP1_TARGET_RANK   d=%d  m=n=%d  r^A=%d  r^v=%d  ell=%d  s = %s\n', ...
        d, m, rA, rv, p, mat2str(SVALS));
fprintf('  threads = %d\n', maxNumCompThreads);
fprintf('  operator %.0f MB,  v %.0f MB,  cores of A*v %.0f MB (rank %d)\n\n', ...
        8*(2*rA + (d-2)*rA^2)*m*n/1e6, 8*(2*rv + (d-2)*rv^2)*n/1e6, ...
        8*n*(2*R + (d-2)*R^2)/1e6, R);

%% ---- the instance, fixed for the whole run ------------------------------
A = rand_tt_operator(d, m, n, rA);
v = rand_tt_vector_decay(d, n, rv, DECADES, REF);

Yall = cell(1,d);
for k = 1:d, Yall{k} = randn(m, max(SVALS) + p); end

exact = A*v;
nex   = norm(exact);

warm_up(WARMUP);

%% ---- anchor -------------------------------------------------------------
anchor = @() anchor_reading(d, 256, 4, 30, 8, p, REPS, TARGET);
anchor_reading(d, 256, 4, 30, 8, p, 2, 0.1);
anc = anchor();

H = {'s', 't_round', 't_seq', 't_fused', 't_pre', 't_src', 't_src_abs', ...
     'e_round', 'e_seq', 'e_fused', 'e_pre', 'e_src', 'e_floor'};
T = nan(numel(SVALS), numel(H));

for ii = 1:numel(SVALS)
    s = SVALS(ii);  l = s + p;
    fprintf('\n=== s = %d   (s/r^v = %.2f,  s(s+ell) = %d,  2 r^v s = %d,  s+2s^2 = %d) ===\n\n', ...
            s, s/rv, s*l, 2*rv*s, s+2*s*s);

    Y    = cell(1,d);  for k = 1:d, Y{k}    = Yall{k}(:, 1:l); end
    Ysrc = cell(1,d);  for k = 1:d, Ysrc{k} = Yall{k}(:, 1:s); end

    % every route builds, inside its own timed call, whatever it needs
    f = { @() round(A*v, 1e-14, s)
          @() stta_matvec_sequential(A, v, s, p, Y)
          @() stta_matvec_fused(A, v, s, p, Y)
          @() stta_matvec_presketched(A, s, p, Y, v)                    % preparation inside
          @() src_matvec_standard(A, v, s, Ysrc)   % raw cores
          @() src_matvec_presketched(A, v, s, Ysrc) };  % pre-sketched

    % ---- accuracy: all routes against A*v, and the best rank-s error ----
    e = nan(1,6);
    e(6) = norm(round(exact, 1e-14, s) - exact)/nex;
    for q = 1:5
        e(q) = norm(f{q}() - exact)/nex;
    end
    fprintf('  floor %.2e | round %.2e  seq %.2e  fused %.2e  pre %.2e  SRC %.2e\n\n', ...
            e(6), e(1), e(2), e(3), e(4), e(5));

    % ---- timing, round-robin, bracketed by the anchor --------------------
    a0 = anchor();
    [t, sp] = time_best_multi(f, REPS, TARGET);
    a1 = anchor();
    anc(end+1) = a0;  anc(end+1) = a1; %#ok<AGROW>

    for q = 1:6
        fprintf('    %-14s %9.4f s  %7.2fx   spread %4.2f\n', ...
                LAB{q}, t(q), t(q)/t(4), sp(q));
    end
    br = max(a0,a1)/min(a0,a1);
    fprintf('  bracket %.4f / %.4f = %.2fx -> %s\n', a0, a1, br, ...
            ternary(br < 1.15, 'this block is usable', 'this block DRIFTED'));

    if exist('ops_model', 'file') == 2
        C = ops_model(d, m, n, rA, rv, s, p);
        fprintf('  counts: fused/ours %.2f   SRC/ours %.2f   absorbed/ours %.2f\n', ...
                C.fused/C.ours, C.src_raw/C.ours, C.src_absorb/C.ours);
    end

    T(ii,:) = [s t e];
end
anc(end+1) = anchor();

write_csv('exp1_target_rank.csv', H, T);

%% ---- the figure's input -------------------------------------------------
% One row per route, one column per s: the time of the route divided by the
% time of the pre-sketched order at that s.
fid = fopen('exp1_target_rank_plot.csv', 'w');
fprintf(fid, 'route');
fprintf(fid, ',s%d', SVALS);
fprintf(fid, '\n');
for q = 1:6
    fprintf(fid, '%s', TAG{q});
    fprintf(fid, ',%.4f', T(:, q+1)./T(:, 5));
    fprintf(fid, '\n');
end
fclose(fid);
fprintf('  written exp1_target_rank_plot.csv\n');

%% ---- summary ------------------------------------------------------------
fprintf('\n  %-14s', 'route');  fprintf('   s = %-4d', SVALS);  fprintf('   (times ours)\n');
fprintf('  %s\n', repmat('-', 1, 50));
for q = 1:6
    fprintf('  %-14s', LAB{q});  fprintf('  %8.2fx', T(:,q+1)./T(:,5));  fprintf('\n');
end
fprintf('\n  anchor readings: '); fprintf('%.4f ', anc);
fprintf('\n  global spread %.2fx; judge each block by its own brackets above.\n', ...
        max(anc)/min(anc));

%% ---- parameters and object sizes ----------------------------------------
fid = fopen('exp1_target_rank.txt', 'w');
fprintf(fid, 'exp1_target_rank\n');
fprintf(fid, 'd %d\nm %d\nn %d\nrA %d\nrv %d\nell %d\nsvals %s\n', ...
        d, m, n, rA, rv, p, mat2str(SVALS));
fprintf(fid, 'decades %g\nref %d\n', DECADES, REF);
fprintf(fid, 'reps %d\ntarget %g\n', REPS, TARGET);
fprintf(fid, 'anchor %s\nanchor_spread %.4f\n', mat2str(anc, 4), max(anc)/min(anc));
fprintf(fid, 'matlab %s\nthreads %d\ncomputer %s\n', version, maxNumCompThreads, computer);
fprintf(fid, 'date %s\n', datestr(now, 31));
fprintf(fid, '\nobject sizes, megabytes, double precision\n');
fprintf(fid, 'operator      %.1f\n', 8*(2*rA + (d-2)*rA^2)*m*n/1e6);
fprintf(fid, 'v             %.1f\n', 8*(2*rv + (d-2)*rv^2)*n/1e6);
fprintf(fid, 'cores of A*v  %.1f\n', 8*n*(2*R + (d-2)*R^2)/1e6);
for s = SVALS
    fprintf(fid, 'A sketched, s = %-3d  %.1f\n', s, 8*(2*rA + (d-2)*rA^2)*m*(s+p)/1e6);
end
fclose(fid);
fprintf('  written exp1_target_rank.txt\n\n');

%% ========================================================================
function t = anchor_reading(d, m, rA, rv, s, p, reps, target)
    st = rng;  rng(12345);
    cleanup = onCleanup(@() rng(st)); %#ok<NASGU>
    A = rand_tt_operator(d, m, m, rA);
    v = rand_tt_vector_decay(d, m, rv, 8, 16);
    Y = cell(1,d);  for k = 1:d, Y{k} = randn(m, s+p); end
    AY = stta_sketch_operator(A, s, p, Y);
    t = time_best(@() stta_matvec_presketched_reused(A, AY, v, p), reps, target);
end

function s = ternary(c, a, b)
    if c, s = a; else, s = b; end
end

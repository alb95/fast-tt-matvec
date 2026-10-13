function t = time_best_multi(fs, reps, target)
%TIME_BEST_MULTI  Time several routines round-robin, so that drift cancels.
%
%   t = time_best_multi(fs, reps, target)
%
%   FS is a cell array of functions; t(q) is the minimum time per call of
%   fs{q}.  Empty entries are skipped and return NaN, and a routine that
%   throws -- running out of memory, typically -- returns NaN without taking
%   the others down with it.
%
%   WHY ROUND-ROBIN.  time_best finishes every repetition of one routine
%   before starting the next, so on a run lasting minutes the first and the
%   last routine are measured on a processor in different states: it starts
%   at its boost clock and settles towards its base clock as the run heats
%   it up.  Closing other applications does not help, because the other
%   application is the benchmark itself.  Measured drift over one of these
%   scripts was 1.34x, which is larger than several of the ratios it exists
%   to establish.
%
%   Here each pass times every routine once, and the passes are repeated.
%   Two readings to be compared are then seconds apart rather than minutes,
%   so a slow thermal drift multiplies them all by nearly the same factor and
%   cancels in the ratio.  The minimum over passes is still taken, because
%   the noise on a shared machine is one-sided: a call can be delayed, never
%   accelerated.
%
%   The number of inner calls is calibrated once per routine, before the
%   passes begin, so that each timed block lasts at least TARGET seconds and
%   the timer resolution never enters.
%
%   The absolute times are still affected by the drift, and should be read as
%   what they are: the ratios are the result, the seconds are context.

if nargin < 2 || isempty(reps),   reps   = 3;    end
if nargin < 3 || isempty(target), target = 0.25; end

nf  = numel(fs);
nin = ones(1, nf);
ok  = false(1, nf);

% ---- calibration, and a first chance for a routine to fail ---------------
for q = 1:nf
    if isempty(fs{q}), continue; end
    try
        fs{q}();                                   % warm up, not timed
        tic; fs{q}(); t0 = toc;
        nin(q) = max(1, ceil(target/max(t0, 1e-6)));
        ok(q)  = true;
    catch ME
        fprintf('    routine %d did not run: %s\n', q, ME.message);
    end
end

% ---- the passes ----------------------------------------------------------
t = nan(1, nf);
t(ok) = inf;
for b = 1:reps
    for q = 1:nf
        if ~ok(q), continue; end
        try
            tic
            for i = 1:nin(q), fs{q}(); end
            t(q) = min(t(q), toc/nin(q));
        catch ME
            fprintf('    routine %d failed on pass %d: %s\n', q, b, ME.message);
            ok(q) = false;  t(q) = NaN;
        end
    end
end
end
function [t, spread] = time_best(f, reps, target)
%TIME_BEST  Seconds per call of f, minimum over REPS blocks.
%
%   [t, spread] = time_best(f, reps, target)
%
%   The block length is calibrated so that each block lasts at least TARGET
%   seconds, which keeps the timer resolution and the per-call overhead out
%   of the result, and the minimum over blocks is reported because on a
%   shared machine the noise is one-sided: a call can be delayed, never
%   accelerated.  The first call is discarded (JIT, cold caches).
%
%   SPREAD is the ratio of the slowest to the fastest reading.  It is the
%   evidence that the minimum can be trusted: near one it is solid, far from
%   one the readings are not measuring the same work and the cause has to be
%   found before any of them is used.  It is a diagnostic, not a rule --
%   nothing is discarded here.

if nargin < 2, reps   = 5;    end
if nargin < 3, target = 0.25; end

f();                                   % warm up, not timed
tic; f(); t0 = toc;
nin = max(1, ceil(target / max(t0, 1e-6)));

t = inf;  tmax = 0;
for b = 1:reps
    tic
    for q = 1:nin, f(); end
    tb   = toc/nin;
    t    = min(t, tb);
    tmax = max(tmax, tb);
end
spread = tmax / t;
end
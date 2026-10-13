function t = time_best(f, reps, target)
%TIME_BEST  Seconds per call of f, minimum over REPS blocks.
%
%   t = time_best(f, reps, target)
%
%   The block length is calibrated so that each block lasts at least TARGET
%   seconds, which keeps the timer resolution and the per-call overhead out
%   of the result, and the minimum over blocks is reported because on a
%   shared machine the noise is one-sided: a call can be delayed, never
%   accelerated.  The first call is discarded (JIT, cold caches).

if nargin < 2, reps   = 5;    end
if nargin < 3, target = 0.25; end

f();                                   % warm up, not timed
tic; f(); t0 = toc;
nin = max(1, ceil(target / max(t0, 1e-6)));

t = inf;
for b = 1:reps
    tic
    for q = 1:nin, f(); end
    t = min(t, toc/nin);
end
end
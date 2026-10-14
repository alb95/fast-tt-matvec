function warm_up(seconds)
%WARM_UP  Hold the processor at its steady clock before any timing starts.
%
%   warm_up(seconds)
%
%   WHY.  A laptop processor starts a run at its boost clock and settles
%   towards its base clock as the run heats it up.  The settling is not noise:
%   it is reproducible, and a script launched on a cold machine measures its
%   first routines fast and its last ones slow, by as much as 40 per cent over
%   a few minutes.  Round-robin timing makes that drift common to the routines
%   being compared, so it cancels in the ratios, but the absolute seconds stay
%   contaminated and the per-reading spread stays large.
%
%   Running a dense load first moves the whole measurement past the transient,
%   so that the readings are taken on a processor already at its steady clock.
%   On the machine used for the paper the drift over a run falls from about
%   1.4 to about 1.06 once this is done.
%
%   The default of 90 seconds was enough there; a machine with a larger
%   thermal mass may need more.  Pass 0 to skip, which is what a quick
%   exploratory run wants.
%
%   The state of the random generator is saved and restored, so that calling
%   this changes nothing about the problem the experiment then builds.

if nargin < 1 || isempty(seconds), seconds = 90; end
if seconds <= 0, return; end

% The load below draws from the global generator, so the state is saved and
% restored: a warm-up must not shift the random stream that the experiment is
% seeded to reproduce.
st = rng;
cleanup = onCleanup(@() rng(st)); %#ok<NASGU>

fprintf('  warming up for %g s so that the timings are taken at a steady clock', seconds);
X = randn(2000);
acc = 0;
t0 = tic;
next = 10;
while toc(t0) < seconds
    Y = X*X;
    acc = acc + Y(1,1);                 % keeps the product from being elided
    if toc(t0) > next, fprintf('.'); next = next + 10; end
end
fprintf(' done (%.3g)\n', acc*0);       % acc is used, and prints as 0
end

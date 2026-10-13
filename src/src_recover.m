function w = src_recover(A, v, s, Yd, Lc)
%SRC_RECOVER  The second sweep of SRC, from the first sweep's output.
%
%   w = src_recover(A, v, s, Yd, Lc)
%
%   [Yd, Lc] come from src_sketch.  Together the two routines are
%   src_matvec_standard with a pre-sketched first sweep, split
%   in two so that the first part, which is linear in v, can be accumulated
%   over several arguments.  This is the one-argument case of
%   src_recover_sum, and is implemented by calling it.

w = src_recover_sum(A, {v}, s, {Yd}, {Lc});
end
function w = stta_matvec_presketched_reused(A, AY, v, p)
%STTA_MATVEC_PRESKETCHED_REUSED  STTA on an operator whose cores have been pre-sketched.
%
%   AY = stta_sketch_operator(A, s, p, Y) is built ONCE per
%   operator and passed in: it is not part of the per-product work and is
%   timed separately.  After it, no pass over the operator carries r^v, so
%   the m*n term depends on the sketch width alone.

[Om, Ps] = stta_compress_presketched_reused(A, AY, v, p);
w = stta_recover(Om, Ps);
end
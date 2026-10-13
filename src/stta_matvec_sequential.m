function w = stta_matvec_sequential(A, v, s, p, Y)
%STTA_MATVEC_SEQUENTIAL  STTA in the sequential order: form the cores, then sketch.
%
%   The cores of A*v are built site by site and each is contracted against
%   the sketch as soon as it exists.  Nothing is ever pre-processed, so the
%   pass over the operator carries the rank of the argument: this is the
%   order whose m*n term is (r^v)^2 and therefore the only one of the three
%   that does not depend on the target rank.

[Om, Ps] = stta_compress_sequential(A, v, s, p, Y);
w = stta_recover(Om, Ps);
end
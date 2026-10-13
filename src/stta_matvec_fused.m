function w = stta_matvec_fused(A, v, s, p, Y)
%STTA_MATVEC_FUSED  STTA in the fused order: sketch while traversing the cores.
%
%   The cores of A*v are never formed; the sketch is carried along the
%   traversal.  The operator is swept in its original form, once per sweep,
%   so the m*n term carries both r^v and the sketch width; the assembly of
%   Psi costs no further pass over A, because the right sweep's environment
%   is kept and used twice (see stta_compress_fused).  Weight
%   r^v(2s+ell).

[Om, Ps] = stta_compress_fused(A, v, s, p, Y);
w = stta_recover(Om, Ps);
end
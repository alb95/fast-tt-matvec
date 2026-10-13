function [Yd, Lc] = src_sketch(A, v, s, Y, AY)
%SRC_SKETCH  The first sweep of SRC, alone.
%
%   [Yd, Lc] = src_sketch(A, v, s, Y)
%   [Yd, Lc] = src_sketch(A, v, s, Y, AY)
%
%   Returns everything the second sweep needs and nothing else:
%
%       Yd      [s, n(d)]              the sketch with the last mode free
%       Lc{k}   [s, r_A(k+1), r^v(k+1)]  the left carries, k = 1, ..., d-1
%
%   THIS SWEEP IS LINEAR IN v.  No QR has happened yet, so the outputs of two
%   arguments may be added: that is what lets SRC compress a sum, provided the
%   arguments can be revisited afterwards (see src_recover_sum).
%   The second sweep is not linear, because its projector is computed from a
%   QR of the argument.
%
%   The sweep is pre-sketched: with five arguments AY is the sketched operator
%   of stta_sketch_operator(A, s, 0, Y), built once per operator and
%   reused for every argument; with four it is built here.  Only the first s
%   columns of Y are used -- SRC has no oversampling.

d  = length(v.n);
Am = A.m;  An = A.n;  Ar = A.r;  Aps = A.ps;  Acore = A.core;
vps = v.ps;  vr = v.r;  vn = v.n;  vcore = v.core;
Vcr = @(k) reshape(vcore(vps(k):vps(k+1)-1), [vr(k), vn(k), vr(k+1)]);

Ys = cell(1,d);
for k = 1:d, Ys{k} = Y{k}(:,1:s); end
if nargin < 5 || isempty(AY)
    AY = stta_sketch_operator(A, s, 0, Ys);
else
    for k = 1:d, AY{k} = AY{k}(:,:,:,1:s); end
end

Lc = cell(1,d);
T  = tensorprod(reshape(AY{1}, [Ar(2), Am(1), s]), ...
    reshape(Vcr(1), [vn(1), vr(2)]), 2, 1);
Lc{1} = permute(reshape(T, [Ar(2), s, vr(2)]), [2 1 3]);

for k = 2:d-1
    Lp = permute(Lc{k-1}, [2 3 1]);              % [Ar(k), vr(k), s]
    Vm = reshape(Vcr(k), [vr(k), vn(k)*vr(k+1)]);
    T  = pagemtimes(Lp, Vm);
    T  = reshape(T, [Ar(k)*vn(k), vr(k+1), s]);
    M  = permute(AY{k}, [1 3 2 4]);
    M  = reshape(M, [Ar(k)*Am(k), Ar(k+1), s]);
    Lc{k} = permute(pagemtimes(M, 'transpose', T, 'none'), [3 1 2]);
end

% last site: the mode is left free, so this is the matrix the QR will act on
Lp = permute(Lc{d-1}, [2 3 1]);
Vm = reshape(Vcr(d), [vr(d), vn(d)]);
T  = reshape(pagemtimes(Lp, Vm), [Ar(d)*vn(d), 1, s]);
Gd = reshape(permute(reshape(Acore(Aps(d):Aps(d+1)-1), [Ar(d), An(d), Am(d)]), ...
    [1 3 2]), [Ar(d)*Am(d), An(d)]);
Yd = reshape(pagemtimes(Gd, 'transpose', T, 'none'), [An(d), s]).';
end
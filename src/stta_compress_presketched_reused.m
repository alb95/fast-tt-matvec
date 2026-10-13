function [Omega, Psi, varargout] = stta_compress_presketched_reused(A, AY, v, p)
%STTA_COMPRESS_PRESKETCHED_REUSED  Sketches of A*v from the pre-sketched cores.
%
%   [Omega, Psi] = stta_compress_presketched_reused(A, AY, v, p)
%
%   AY comes from stta_sketch_operator.  Neither the cores of A*v
%   nor any part of them is formed.  The two sweeps see the operator only
%   through AY, whose row mode has already been replaced by the sketch
%   dimension; the raw cores of A are needed only in the assembly of Psi,
%   whose k-th row mode is left free.
%
%   Environments:  Rc{k} is [Ar(k), vr(k), s],  Lc{k} is [Ar(k+1), vr(k+1), l].

l  = size(AY{1}, 4);
s  = l - p;
d  = length(v.n);

vps = v.ps;  vr = v.r;  vn = v.n;  vcore = v.core;
Am  = A.m;   An = A.n;  Ar = A.r;  Aps = A.ps;  Acore = A.core;

Vcr = @(k) reshape(vcore(vps(k):vps(k+1)-1), [vr(k), vn(k), vr(k+1)]);
Acr = @(k) reshape(Acore(Aps(k):Aps(k+1)-1), [Ar(k), An(k), Am(k), Ar(k+1)]);

%% right sweep
Rc = cell(1,d);
Z  = tensorprod(reshape(AY{d}(:,1,:,1:s), [Ar(d), Am(d), s]), ...
                reshape(Vcr(d), [vr(d), vn(d)]), 2, 2);      % [Ar(d), s, vr(d)]
Rc{d} = permute(reshape(Z, [Ar(d), s, vr(d)]), [1 3 2]);     % [Ar(d), vr(d), s]

for k = d:-1:2
    W = tensorprod(Vcr(k-1), Rc{k}, 3, 2);                   % [vr(k-1),vn(k-1),Ar(k),s]
    W = reshape(W, [vr(k-1), vn(k-1), Ar(k), s]);
    W = permute(W, [3 2 1 4]);
    W = reshape(W, [Ar(k)*vn(k-1), vr(k-1), s]);
    M = reshape(AY{k-1}(:,:,:,1:s), [Ar(k-1), Ar(k)*Am(k-1), s]);
    Rc{k-1} = pagemtimes(M, W);                              % [Ar(k-1), vr(k-1), s]
end

%% left sweep
Lc = cell(1,d);
Z  = tensorprod(reshape(AY{1}(1,:,:,:), [Ar(2), Am(1), l]), ...
                reshape(Vcr(1), [vn(1), vr(2)]), 2, 1);      % [Ar(2), l, vr(2)]
Lc{1} = permute(reshape(Z, [Ar(2), l, vr(2)]), [1 3 2]);     % [Ar(2), vr(2), l]

for k = 1:d-1
    W = tensorprod(Vcr(k+1), Lc{k}, 1, 2);                   % [vn(k+1),vr(k+2),Ar(k+1),l]
    W = reshape(W, [vn(k+1), vr(k+2), Ar(k+1), l]);
    W = permute(W, [3 1 2 4]);
    W = reshape(W, [Ar(k+1)*vn(k+1), vr(k+2), l]);
    M = permute(AY{k+1}, [1 3 2 4]);                         % [Ar(k+1),Am(k+1),Ar(k+2),l]
    M = reshape(M, [Ar(k+1)*Am(k+1), Ar(k+2), l]);
    Lc{k+1} = pagemtimes(M, 'transpose', W, 'none');         % [Ar(k+2), vr(k+2), l]
end

[Omega, Psi] = stta_assemble(Acr, Vcr, Lc, Rc, Ar, An, Am, vr, vn, s, l, d);
if nargout >= 3, varargout{1} = reshape(Lc{d}, [l,1]); end
end

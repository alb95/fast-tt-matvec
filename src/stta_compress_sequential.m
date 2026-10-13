function [Omega, Psi] = stta_compress_sequential(A, v, s, p, varargin)
%STTA_COMPRESS_SEQUENTIAL  Sketches of A*v by forming its cores.
%
%   [Omega, Psi] = stta_compress_sequential(A, v, s, p)
%   [Omega, Psi] = stta_compress_sequential(A, v, s, p, Y)
%
%   The k-th core of A*v, of TT ranks A.r(k)*v.r(k), is built when needed and
%   discarded at once, so at most one of them is in memory.  It is needed by
%   the right sweep and again by the left sweep, which also assembles Psi;
%   each core is therefore built twice.  This is the route to prefer when the
%   target rank exceeds the TT ranks of v.

l = s + p;
d = length(v.n);

if nargin >= 5 && ~isempty(varargin{1})
    Y = varargin{1};
else
    Y = cell(1,d);
    for k = 1:d, Y{k} = randn(A.n(k), l); end
end

vps = v.ps;  vr = v.r;  vn = v.n;  vcore = v.core;
Am  = A.m;   An = A.n;  Ar = A.r;  Aps = A.ps;  Acore = A.core;
R = double(Ar(:)).*double(vr(:));

    function C = avcore(k)
        G = reshape(Acore(Aps(k):Aps(k+1)-1), [Ar(k), An(k), Am(k), Ar(k+1)]);
        V = reshape(vcore(vps(k):vps(k+1)-1), [vr(k), vn(k), vr(k+1)]);
        C = tensorprod(G, V, 3, 2);
        C = reshape(C, [Ar(k), An(k), Ar(k+1), vr(k), vr(k+1)]);
        C = reshape(permute(C, [1 4 2 3 5]), [R(k), An(k), R(k+1)]);
    end

%% pass 1: right sweep
Rc = cell(1,d);
C  = avcore(d);
Rc{d} = reshape(C, [R(d), An(d)])*Y{d}(:,1:s);
for k = d:-1:2
    C = avcore(k-1);
    W = tensorprod(C, Rc{k}, 3, 1);
    W = reshape(W, [R(k-1), An(k-1), s]);
    Rc{k-1} = reshape(sum(W .* reshape(Y{k-1}(:,1:s), [1, An(k-1), s]), 2), [R(k-1), s]);
end

%% pass 2: left sweep, assembling Psi on the way
Lc  = cell(1,d);
Psi = cell(1,d);

C = avcore(1);
Lc{1}  = reshape(C, [An(1), R(2)]).'*Y{1};
Psi{1} = reshape(C, [An(1), R(2)])*Rc{2};

for k = 1:d-2
    C = avcore(k+1);
    W = tensorprod(C, Lc{k}, 1, 1);
    W = reshape(W, [An(k+1), R(k+2), l]);
    Lc{k+1} = reshape(sum(W .* reshape(Y{k+1}, [An(k+1), 1, l]), 1), [R(k+2), l]);
    Z = tensorprod(W, Rc{k+2}, 2, 1);
    Z = reshape(Z, [An(k+1), l, s]);
    Psi{k+1} = reshape(permute(Z, [2 1 3]), [l, An(k+1)*s]);
end

C = avcore(d);
Psi{d} = Lc{d-1}.'*reshape(C, [R(d), An(d)]);

Omega = cell(1,d-1);
for k = 1:d-1
    Omega{k} = Lc{k}.'*Rc{k+1};
end
end

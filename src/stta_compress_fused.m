function [Omega, Psi] = stta_compress_fused(A, v, s, p, Y)
%STTA_COMPRESS_FUSED  STTA in the fused order, environment shared.
%
%   [Omega, Psi] = stta_compress_fused(A, v, s, p, Y)
%
%   The cores of A*v are never formed: the sketch is carried along the
%   traversal, and the operator is swept in its original form.  Same output
%   as before, one pass over the operator cores fewer.
%
%   WHAT IS REUSED.  In the right sweep the core of A is applied to the core
%   of v against the incoming environment, with the ROW index left free:
%
%       U_k(a, i, b, j) = sum_{m, a'} A_k(a, i, m, a') V_k(b, m, b') R_{k+1}(a', b', j),
%
%   and the sweep then contracts i with the j-th column of Y_k to obtain
%   R_k.  But U_k is also exactly what the assembly of Psi_k needs:
%
%       Psi_k(t, i, j) = sum_{a, b} L_{k-1}(a, b, t) U_k(a, i, b, j).
%
%   The fused order as usually written throws U_k away and rebuilds the
%   contraction against A_k in the assembly, at a cost of (r^A)^2 m n s (s+ell)
%   per core.  Keeping it makes the assembly free of any pass over A, and the
%   weight of the route falls from
%
%       r^v (2s + ell) + s (s + ell)     to     r^v (2s + ell),
%
%   in units of (r^A)^2 m n per core.  This is the same device as the shared
%   environment of SRC, in which one traversal of the core serves both the
%   projector and the environment; the earlier implementation of this file,
%   which rebuilt the assembly, is superseded by it at every parameter.
%
%   THE ORDER OF THE SWEEPS IS WHAT KEEPS THE MEMORY FLAT.  Psi_k needs
%   L_{k-1}, so the LEFT sweep runs first and stores its carries, which are
%   small: r^A r^v (s+ell) numbers per core.  The right sweep then builds U_k,
%   uses it twice and discards it, so only one U_k, of r^A n r^v s numbers,
%   is ever alive.

l = s + p;
d = length(v.n);

if nargin < 5 || isempty(Y)
    Y = cell(1,d);
    for k = 1:d, Y{k} = randn(A.n(k), l); end
end

Am = A.m;  An = A.n;  Ar = A.r;  Aps = A.ps;  Acore = A.core;
vps = v.ps;  vr = v.r;  vn = v.n;  vcore = v.core;
Vcr = @(k) reshape(vcore(vps(k):vps(k+1)-1), [vr(k), vn(k), vr(k+1)]);
Acr = @(k) reshape(Acore(Aps(k):Aps(k+1)-1), [Ar(k), An(k), Am(k), Ar(k+1)]);

%% ---- left sweep, on the raw cores:  Lc{k} is [Ar(k+1), vr(k+1), l] -------
Lc = cell(1,d);

% site 1: Ar(1) = vr(1) = 1
Aleft = reshape(permute(Acr(1), [4 2 1 3]), [Ar(2)*An(1), Am(1)]);
Z = reshape(Aleft * reshape(Vcr(1), [vn(1), vr(2)]), [Ar(2), An(1), vr(2)]);
Lc{1} = reshape(sum(Z .* reshape(Y{1}(:,1:l), [1, An(1), 1, l]), 2), [Ar(2), vr(2), l]);

for k = 2:d-1
    % rows (a', i), columns (a, m): the left bond and the column index are
    % summed, the right bond and the row index stay free
    Aleft = reshape(permute(Acr(k), [4 2 1 3]), [Ar(k+1)*An(k), Ar(k)*Am(k)]);
    W = tensorprod(Vcr(k), Lc{k-1}, 1, 2);                  % [vn, vr', Ar, l]
    W = reshape(W, [vn(k), vr(k+1), Ar(k), l]);
    W = reshape(permute(W, [3 1 2 4]), [Ar(k)*vn(k), vr(k+1)*l]);
    Z = reshape(Aleft * W, [Ar(k+1), An(k), vr(k+1), l]);
    Lc{k} = reshape(sum(Z .* reshape(Y{k}(:,1:l), [1, An(k), 1, l]), 2), ...
                    [Ar(k+1), vr(k+1), l]);
end

%% ---- right sweep, building U_k and using it twice ------------------------
Rc  = cell(1,d);
Psi = cell(1,d);

% site d: the right bond is 1, so U_d carries no sketch index
Adr = reshape(Acr(d), [Ar(d)*An(d), Am(d)]);                % rows (a, i)
U = reshape(Adr * reshape(Vcr(d), [vr(d), vn(d)]).', [Ar(d), An(d), vr(d)]);
Rc{d} = reshape(sum(U .* reshape(Y{d}(:,1:s), [1, An(d), 1, s]), 2), [Ar(d), vr(d), s]);
Psi{d} = reshape(tensorprod(Lc{d-1}, U, [1 2], [1 3]), [l, An(d)]);

for k = d-1:-1:2
    % rows (a, i), columns (m, a'): a plain reshape of the core, no transpose
    Aright = reshape(Acr(k), [Ar(k)*An(k), Am(k)*Ar(k+1)]);
    W = tensorprod(Vcr(k), Rc{k+1}, 3, 2);                  % [vr, vn, Ar', s]
    W = reshape(W, [vr(k), vn(k), Ar(k+1), s]);
    W = reshape(permute(W, [2 3 1 4]), [vn(k)*Ar(k+1), vr(k)*s]);
    U = reshape(Aright * W, [Ar(k), An(k), vr(k), s]);      % the shared environment
    Rc{k}  = reshape(sum(U .* reshape(Y{k}(:,1:s), [1, An(k), 1, s]), 2), ...
                     [Ar(k), vr(k), s]);
    Psi{k} = reshape(tensorprod(Lc{k-1}, U, [1 2], [1 3]), [l, An(k)*s]);
end

% site 1: Ar(1) = vr(1) = 1, so U_1 is Psi_1 itself and R_1 is not needed
W = tensorprod(Vcr(1), Rc{2}, 3, 2);
W = reshape(W, [vn(1)*Ar(2), s]);
Psi{1} = reshape(Acr(1), [An(1), Am(1)*Ar(2)]) * W;

%% ---- Omega ---------------------------------------------------------------
Omega = cell(1,d-1);
for k = 1:d-1
    Omega{k} = reshape(tensorprod(Lc{k}, Rc{k+1}, [1 2], [1 2]), [l, s]);
end
end
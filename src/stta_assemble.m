function [Omega, Psi] = stta_assemble(Acr, Vcr, Lc, Rc, Ar, An, Am, vr, vn, s, l, d)
%STTA_ASSEMBLE  Omega and Psi from the two sweeps.
%
%   Shared by every route that does not form the cores of A*v.  Omega{k} is a
%   contraction of the two environments alone.  Psi{k} leaves the k-th row
%   mode free, so it is the one place where the original core of A is still
%   traversed with both of its mode indices.
%
%   BLOCKED OVER THE SKETCH COLUMN.  The flop-optimal contraction order for a
%   middle site contracts the left environment before the operator, which
%   materialises an intermediate of shape
%
%       [ Ar(k), l, vn(k), Ar(k+1), s ],
%
%   of size r_A^2 n l s.  That is harmless while r_A is small and ruinous when
%   it is not: at r_A = 110, n = 96, s = 24, l = 28 it is 6.2 GB per site, and
%   the routine becomes memory-bound long before it becomes compute-bound.
%   Measured on the near-resonant solve, the pre-sketched order was doing 2.5
%   times FEWER operations than SRC and taking 2.3 times longer -- a factor
%   5.7 in effective throughput, all of it traffic on that array.
%
%   The sketch column is a batch index, so the same contraction can be done in
%   blocks of columns at identical operation count and bounded memory.  The
%   block width is chosen so that the live intermediate stays under MAXBYTES,
%   and is at least one column; the operator core is permuted into a matrix
%   once, outside the loop, rather than by tensorprod at every block.

MAXBYTES = 256e6;

Omega = cell(1,d-1);
for k = 1:d-1
    Omega{k} = tensorprod(Lc{k}, Rc{k+1}, [1 2], [1 2]);
    Omega{k} = reshape(Omega{k}, [l, s]);
end

Psi = cell(1,d);

W = tensorprod(Vcr(1), Rc{2}, 3, 2);
W = reshape(W, [vr(1), vn(1), Ar(2), s]);
Z = tensorprod(Acr(1), W, [3 4], [2 3]);
Psi{1} = reshape(Z, [An(1), s]);

for k = 2:d-1
    % small: [vr(k), vn(k), Ar(k+1), s], built once
    W0 = tensorprod(Vcr(k), Rc{k+1}, 3, 2);
    W0 = reshape(W0, [vr(k), vn(k), Ar(k+1), s]);

    % the operator core as a matrix, An(k) x (Ar(k)*Am(k)*Ar(k+1)), once
    Ak = reshape(permute(Acr(k), [2 1 3 4]), ...
                 [An(k), Ar(k)*Am(k)*Ar(k+1)]);

    per = 8*Ar(k)*vn(k)*Ar(k+1)*s;            % bytes of one sketch column
    nb  = max(1, min(l, floor(MAXBYTES/max(per,1))));

    Lk = Lc{k-1};                              % [Ar(k), vr(k), l]
    Z  = zeros(An(k), l, s);
    for t0 = 1:nb:l
        t1 = min(t0+nb-1, l);  nt = t1 - t0 + 1;
        Wt = tensorprod(Lk(:,:,t0:t1), W0, 2, 1);          % [Ar,nt,vn,Ar',s]
        Wt = reshape(Wt, [Ar(k), nt, vn(k), Ar(k+1), s]);
        Wt = permute(Wt, [1 3 4 5 2]);                     % [Ar,vn,Ar',s,nt]
        Wt = reshape(Wt, [Ar(k)*vn(k)*Ar(k+1), s*nt]);
        Zt = reshape(Ak*Wt, [An(k), s, nt]);
        Z(:,t0:t1,:) = permute(Zt, [1 3 2]);
    end
    Psi{k} = reshape(permute(Z, [2 1 3]), [l, An(k)*s]);
end

W = tensorprod(Lc{d-1}, Vcr(d), 2, 1);
W = reshape(W, [Ar(d), l, vn(d), vr(d+1)]);
Z = tensorprod(W, Acr(d), [1 3], [1 3]);
Psi{d} = reshape(Z, [l, An(d)]);
end
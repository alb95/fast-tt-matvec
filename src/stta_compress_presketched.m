function [Omega, Psi] = stta_compress_presketched(P, v)
%STTA_COMPRESS_PRESKETCHED  The pre-sketched route on a prepared operator.
%
%   [Omega, Psi] = stta_compress_presketched(P, v)
%
%   P comes from stta_prepare_presketched.  Same algorithm, same output and
%   the same operation count as stta_compress_presketched_reused: the only
%   difference is that every array layout depending on the operator alone is
%   read from P instead of being rebuilt here.
%
%   The operator is not an argument.  After the preparation this route never
%   reads A again, so the work split is explicit: stta_prepare_presketched
%   does everything that depends on A alone, this file everything that
%   depends on v.  In the experiments both are called inside the timed
%   region, so the preparation is charged to every product.

d  = P.d;
s  = P.s;
l  = P.l;
Ar = P.Ar;  An = P.An;  Am = P.Am;

vps = v.ps;  vr = v.r;  vn = v.n;  vcore = v.core;
Vcr = @(k) reshape(vcore(vps(k):vps(k+1)-1), [vr(k), vn(k), vr(k+1)]);

%% right sweep
Rc = cell(1,d);
Z  = tensorprod(P.AYd, reshape(Vcr(d), [vr(d), vn(d)]), 2, 2);   % [Ar(d), s, vr(d)]
Rc{d} = permute(reshape(Z, [Ar(d), s, vr(d)]), [1 3 2]);         % [Ar(d), vr(d), s]

for k = d:-1:2
    W = tensorprod(Vcr(k-1), Rc{k}, 3, 2);                       % [vr,vn,Ar(k),s]
    W = reshape(W, [vr(k-1), vn(k-1), Ar(k), s]);
    W = permute(W, [3 2 1 4]);
    W = reshape(W, [Ar(k)*vn(k-1), vr(k-1), s]);
    Rc{k-1} = pagemtimes(P.AYr{k-1}, W);                         % [Ar(k-1), vr(k-1), s]
end

%% left sweep
Lc = cell(1,d);
Z  = tensorprod(P.AY1, reshape(Vcr(1), [vn(1), vr(2)]), 2, 1);   % [Ar(2), l, vr(2)]
Lc{1} = permute(reshape(Z, [Ar(2), l, vr(2)]), [1 3 2]);         % [Ar(2), vr(2), l]

for k = 1:d-1
    W = tensorprod(Vcr(k+1), Lc{k}, 1, 2);                       % [vn,vr,Ar(k+1),l]
    W = reshape(W, [vn(k+1), vr(k+2), Ar(k+1), l]);
    W = permute(W, [3 1 2 4]);
    W = reshape(W, [Ar(k+1)*vn(k+1), vr(k+2), l]);
    Lc{k+1} = pagemtimes(P.AYl{k+1}, 'transpose', W, 'none');    % [Ar(k+2),vr(k+2),l]
end

%% assembly
% Identical to stta_assemble, with the three operator layouts read
% from P.  The blocking over the sketch column is kept: the sketch index is a
% batch index, so the contraction can be done in blocks of columns at the same
% operation count and bounded working memory.
MAXBYTES = 256e6;

Omega = cell(1,d-1);
for k = 1:d-1
    Omega{k} = reshape(tensorprod(Lc{k}, Rc{k+1}, [1 2], [1 2]), [l, s]);
end

Psi = cell(1,d);

% first site: A1 is [An(1), Am(1)*Ar(2)] and W is [vn(1), Ar(2), s] with
% vr(1) = 1, so both group contiguously and no permute is needed
W = tensorprod(Vcr(1), Rc{2}, 3, 2);
W = reshape(W, [vn(1)*Ar(2), s]);
Psi{1} = P.A1 * W;

for k = 2:d-1
    W0 = tensorprod(Vcr(k), Rc{k+1}, 3, 2);
    W0 = reshape(W0, [vr(k), vn(k), Ar(k+1), s]);

    per = 8*Ar(k)*vn(k)*Ar(k+1)*s;             % bytes of one sketch column
    nb  = max(1, min(l, floor(MAXBYTES/max(per,1))));

    Lk = Lc{k-1};                              % [Ar(k), vr(k), l]
    Z  = zeros(An(k), l, s);
    for t0 = 1:nb:l
        t1 = min(t0+nb-1, l);  nt = t1 - t0 + 1;
        Wt = tensorprod(Lk(:,:,t0:t1), W0, 2, 1);          % [Ar,nt,vn,Ar',s]
        Wt = reshape(Wt, [Ar(k), nt, vn(k), Ar(k+1), s]);
        Wt = permute(Wt, [1 3 4 5 2]);                     % [Ar,vn,Ar',s,nt]
        Wt = reshape(Wt, [Ar(k)*vn(k)*Ar(k+1), s*nt]);
        Zt = reshape(P.Aasm{k}*Wt, [An(k), s, nt]);
        Z(:,t0:t1,:) = permute(Zt, [1 3 2]);
    end
    Psi{k} = reshape(permute(Z, [2 1 3]), [l, An(k)*s]);
end

% last site: Ad is [Ar(d)*Am(d), An(d)]
W = tensorprod(Lc{d-1}, Vcr(d), 2, 1);
W = reshape(W, [Ar(d), l, vn(d)]);
W = reshape(permute(W, [2 1 3]), [l, Ar(d)*vn(d)]);
Psi{d} = W * P.Ad;
end
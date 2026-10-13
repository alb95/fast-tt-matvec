function [Omega, Psi, varargout] = stta_sketch(v, s, p, varargin)
%STTA_SKETCH  Khatri-Rao STTA sketches of a TT tensor.
%
%   [Omega, Psi] = stta_sketch(v, s, p)
%   [Omega, Psi] = stta_sketch(v, s, p, Y)
%   [Omega, Psi, Sv] = ...   also returns the Khatri-Rao sketch Y_{<=d}' v
%
%   Y{k} has size v.n(k) x (s+p).  The sketch column index is a batch index:
%   the two sweeps are therefore batched matrix products over that index.

l = s + p;
d = length(v.n);

if nargin >= 4 && ~isempty(varargin{1})
    Y = varargin{1};
else
    Y = cell(1,d);
    for k = 1:d, Y{k} = randn(v.n(k), l); end
end

vps = v.ps;  vr = v.r;  vn = v.n;  vcore = v.core;
Vcr = @(k) reshape(vcore(vps(k):vps(k+1)-1), [vr(k), vn(k), vr(k+1)]);

% vY{k}(c,t,e) = sum_j V{k}(c,j,e) Y{k}(j,t)
vY = cell(1,d);
for k = 1:d
    Z = tensorprod(Vcr(k), Y{k}, 2, 1);                 % [vr(k), vr(k+1), l]
    Z = reshape(Z, [vr(k), vr(k+1), l]);
    vY{k} = permute(Z, [1 3 2]);                        % [vr(k), l, vr(k+1)]
end

%% right sweep, batched over the sketch column
Rc = cell(1,d);
Rc{d} = reshape(vY{d}(:,1:s,1), [vr(d), s]);
for k = d:-1:2
    P = permute(vY{k-1}(:,1:s,:), [1 3 2]);             % [vr(k-1), vr(k), s]
    X = reshape(Rc{k}, [vr(k), 1, s]);
    Rc{k-1} = reshape(pagemtimes(P, X), [vr(k-1), s]);
end

%% left sweep
Lc = cell(1,d);
Lc{1} = reshape(vY{1}, [l, vr(2)]).';                   % [vr(2), l]
for k = 1:d-1
    P = permute(vY{k+1}, [1 3 2]);                      % [vr(k+1), vr(k+2), l]
    X = reshape(Lc{k}, [vr(k+1), 1, l]);
    Lc{k+1} = reshape(pagemtimes(P, 'transpose', X, 'none'), [vr(k+2), l]);
end

%% assembly
Omega = cell(1,d-1);
for k = 1:d-1
    Omega{k} = Lc{k}.'*Rc{k+1};
end

Psi = cell(1,d);
vX = cell(1,d-1);
for k = 1:d-1
    vX{k} = reshape(Vcr(k), [vr(k)*vn(k), vr(k+1)])*Rc{k+1};
end
Psi{1} = vX{1};
for k = 2:d-1
    Psi{k} = Lc{k-1}.'*reshape(vX{k}, [vr(k), vn(k)*s]);
end
Psi{d} = Lc{d-1}.'*reshape(Vcr(d), [vr(d), vn(d)]);

if nargout >= 3, varargout{1} = reshape(Lc{d}, [l,1]); end
end

function [AY, varargout] = stta_sketch_operator(A, s, p, varargin)
%STTA_SKETCH_OPERATOR  Absorb the sketch into the cores of A.
%
%   AY = stta_sketch_operator(A, s, p)
%   AY = stta_sketch_operator(A, s, p, Y)
%
%   Y{k} has size A.n(k) x (s+p) and acts on the ROW mode of A.
%   AY{k} has size [A.r(k), A.r(k+1), A.m(k), s+p]; the row mode has been
%   replaced by the sketch dimension.  This is the only step that depends on
%   A alone, and is therefore performed once per operator.

l = s + p;
d = length(A.n);

if nargin >= 4 && ~isempty(varargin{1})
    Y = varargin{1};
else
    Y = cell(1,d);
    for k = 1:d, Y{k} = randn(A.n(k), l); end
end

Am = A.m;  An = A.n;  Ar = A.r;  Aps = A.ps;  Acore = A.core;

AY = cell(1,d);
for k = 1:d
    G = reshape(Acore(Aps(k):Aps(k+1)-1), [Ar(k), An(k), Am(k), Ar(k+1)]);
    G = permute(G, [1 4 3 2]);                              % [r, r', m, n]
    G = reshape(G, [Ar(k)*Ar(k+1)*Am(k), An(k)]);
    AY{k} = reshape(G*Y{k}, [Ar(k), Ar(k+1), Am(k), l]);
end

if nargout >= 2, varargout{1} = Y; end
end

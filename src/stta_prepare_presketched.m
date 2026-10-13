function P = stta_prepare_presketched(A, s, p, Y)
%STTA_PREPARE_PRESKETCHED  The pre-sketch, and every operator layout, in one pass.
%
%   P = stta_prepare_presketched(A, s, p, Y)
%
%   Everything the pre-sketched route needs that depends on the operator
%   alone: the sketched operator, in the two layouts its two sweeps read, and
%   the raw cores in the layout its assembly reads.
%   stta_compress_presketched(P, v) then performs a product without touching
%   A again.
%
%   WHY ONE PASS.  The route used to transpose every operator core twice per
%   product, in two different layouts:
%
%       stta_sketch_operator   permute(G, [1 4 3 2]), to contract
%                                         the row mode n against Y;
%       stta_assemble          permute(G, [2 1 3 4]), to leave the
%                                         row mode n free.
%
%   Both separate the same index n.  So the core is transposed ONCE, into
%
%       Ak = [ n ,  r m r' ],
%
%   and used twice: transposed for the sketch, Ak' * Y, which MATLAB hands to
%   a transposed gemm without forming Ak', and as it is for the assembly.  The
%   layouts of the sketched operator that the two sweeps read come out of the
%   sketch by a free reshape (left sweep) and by a permute of an array
%   smaller than the core by a factor n / (s + ell) (right sweep).
%
%   Per product this halves the traffic on the operator cores -- one
%   extraction from A.core and one transposition each, instead of two of each
%   -- and it does so with the preparation still inside the product: this
%   routine is meant to be CALLED WITHIN THE TIMED REGION, exactly like the
%   pre-sketch it replaces.  When the same operator is applied to many
%   right-hand sides it can be called once instead, and that is the saving
%   the paper describes; the experiments charge it every time.
%
%   The values are exactly those of stta_sketch_operator; only
%   their layout differs.  check_prepared.m verifies it entry by entry.

l = s + p;
d = length(A.n);

if nargin < 4 || isempty(Y)
    Y = cell(1,d);
    for k = 1:d, Y{k} = randn(A.n(k), l); end
end

Am = A.m;  An = A.n;  Ar = A.r;  Aps = A.ps;  Acore = A.core;

P = struct('d', d, 's', s, 'p', p, 'l', l, 'Ar', Ar, 'An', An, 'Am', Am);
P.AYr  = cell(1,d);          % right sweep:  [Ar(k), Ar(k+1)*Am(k), s]
P.AYl  = cell(1,d);          % left sweep:   [Ar(k)*Am(k), Ar(k+1), l]
P.Aasm = cell(1,d);          % assembly:     [An(k), Ar(k)*Am(k)*Ar(k+1)]

%% ---- first core: its left bond is 1, so no transposition is needed -----
% [1, An, Am, Ar2] is already the matrix [An, Am*Ar2].
G1   = reshape(Acore(Aps(1):Aps(2)-1), [An(1), Am(1)*Ar(2)]);
P.A1 = G1;                                              % assembly, site 1
X    = reshape(G1' * Y{1}(:,1:l), [Am(1), Ar(2), l]);   % the sketch
P.AY1    = permute(X, [2 1 3]);                         % [Ar(2), Am(1), l]
P.AYr{1} = reshape(P.AY1(:,:,1:s), [1, Ar(2)*Am(1), s]);

%% ---- interior cores: one transposition, used twice ----------------------
for k = 2:d-1
    G  = reshape(Acore(Aps(k):Aps(k+1)-1), [Ar(k), An(k), Am(k), Ar(k+1)]);
    Ak = reshape(permute(G, [2 1 3 4]), [An(k), Ar(k)*Am(k)*Ar(k+1)]);
    P.Aasm{k} = Ak;                                     % assembly
    % the sketch, in layout [Ar, Am, Ar', l]
    X = reshape(Ak' * Y{k}(:,1:l), [Ar(k), Am(k), Ar(k+1), l]);
    P.AYl{k} = reshape(X, [Ar(k)*Am(k), Ar(k+1), l]);   % free
    P.AYr{k} = reshape(permute(X(:,:,:,1:s), [1 3 2 4]), ...
        [Ar(k), Ar(k+1)*Am(k), s]);      % small
end

%% ---- last core: its right bond is 1 -------------------------------------
Gd   = reshape(Acore(Aps(d):Aps(d+1)-1), [Ar(d), An(d), Am(d)]);
P.Ad = reshape(permute(Gd, [1 3 2]), [Ar(d)*Am(d), An(d)]);   % assembly
X    = reshape(P.Ad * Y{d}(:,1:l), [Ar(d), Am(d), l]);        % the sketch
P.AYd    = X(:,:,1:s);                                        % right sweep
P.AYl{d} = reshape(X, [Ar(d)*Am(d), 1, l]);                   % left sweep
end
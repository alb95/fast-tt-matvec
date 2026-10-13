function w = src_matvec_presketched(A, v, s, varargin)
%SRC_MATVEC_PRESKETCHED  SRC, pre-sketched first sweep and
%   absorbed second sweep, each operator core transposed once per product.
%
%   w = src_matvec_presketched(A, v, s)
%   w = src_matvec_presketched(A, v, s, Y)
%   w = src_matvec_presketched(A, v, s, Y, AY)
%
%   Same algorithm and same output as src_matvec_standard_fast;
%   this is SRC done entirely in the pre-sketched style.
%
%   FIRST SWEEP, PRE-SKETCHED.  The operator core is contracted with the
%   random matrix alone, A_k' * Y{k}, of weight r_A^2 n m s, and the sweep
%   then only touches the small sketched cores.
%
%   SECOND SWEEP, ABSORBED.  At each interior site the core is applied to the
%   merged environments to form the matrix whose QR is taken, and its
%   transpose is applied to the orthonormal factor: weight 2 r_A^2 n^2 s^2,
%   against r_A^2 n^2 r^v s for the shared environment.
%
%   ONE TRANSPOSITION PER CORE.  The sketch and both contractions of the
%   second sweep want the same matricisation of the core,
%
%       Ak = [ n ,  r m r' ],
%
%   the sketch as Ak' * Y{k}, the second sweep as Ak * Qq and Ak' * Qk.  It is
%   formed once per core, at the start, and used three times -- the treatment
%   stta_prepare_presketched gives the pre-sketched STTA route, so the two
%   remain comparable.  MATLAB dispatches Ak' * X to a transposed gemm without
%   forming Ak'.  The two edge cores need no transposition beyond a reshape
%   (first) or a single permute (last), each also used for everything.
%
%   All of it happens inside the call: a timing of this routine charges the
%   pre-sketch and the transpositions to the product.  With five arguments the
%   sketched operator AY, in the layout of stta_sketch_operator,
%   is used for the first sweep instead of being computed here.
%
%   Only the first s columns of Y are used; SRC has no oversampling.

d  = length(v.n);
Am = A.m;  An = A.n;  Ar = A.r;  Aps = A.ps;  Acore = A.core;
vps = v.ps;  vr = v.r;  vn = v.n;  vcore = v.core;

Vcr = @(k) reshape(vcore(vps(k):vps(k+1)-1), [vr(k), vn(k), vr(k+1)]);

if nargin >= 4 && ~isempty(varargin{1})
    Y = varargin{1};
else
    Y = cell(1,d);
    for k = 1:d, Y{k} = randn(An(k), s); end
end
Ys = cell(1,d);
for k = 1:d, Ys{k} = Y{k}(:,1:s); end
haveAY = nargin >= 5 && ~isempty(varargin{2});

%% ---- the operator, one transposition per core --------------------------
% G1  [An(1), Am(1)*Ar(2)]                 first core, a reshape
% Ak  [An(k), Ar(k)*Am(k)*Ar(k+1)]         interior cores
% Gd  [Ar(d)*Am(d), An(d)]                 last core
G1 = reshape(Acore(Aps(1):Aps(2)-1), [An(1), Am(1)*Ar(2)]);
Ak = cell(1,d);
for k = 2:d-1
    G = reshape(Acore(Aps(k):Aps(k+1)-1), [Ar(k), An(k), Am(k), Ar(k+1)]);
    Ak{k} = reshape(permute(G, [2 1 3 4]), [An(k), Ar(k)*Am(k)*Ar(k+1)]);
end
Gd = reshape(Acore(Aps(d):Aps(d+1)-1), [Ar(d), An(d), Am(d)]);
Gd = reshape(permute(Gd, [1 3 2]), [Ar(d)*Am(d), An(d)]);

%% ---- first sweep, pre-sketched ------------------------------------------
Lc = cell(1,d);                                  % Lc{k} is [s, Ar(k+1), vr(k+1)]

if haveAY
    X = permute(reshape(varargin{2}{1}(:,:,:,1:s), [Ar(2), Am(1), s]), [2 1 3]);
else
    X = reshape(G1' * Ys{1}, [Am(1), Ar(2), s]);                % the sketch
end
T  = tensorprod(X, reshape(Vcr(1), [vn(1), vr(2)]), 1, 1);     % [Ar(2), s, vr(2)]
Lc{1} = permute(reshape(T, [Ar(2), s, vr(2)]), [2 1 3]);

for k = 2:d-1
    if haveAY
        M = permute(varargin{2}{k}(:,:,:,1:s), [1 3 2 4]);
    else
        M = Ak{k}' * Ys{k};                                     % the sketch
    end
    M  = reshape(M, [Ar(k)*Am(k), Ar(k+1), s]);
    Lp = permute(Lc{k-1}, [2 3 1]);                             % [Ar(k), vr(k), s]
    Vm = reshape(Vcr(k), [vr(k), vn(k)*vr(k+1)]);
    T  = pagemtimes(Lp, Vm);
    T  = reshape(T, [Ar(k)*vn(k), vr(k+1), s]);
    Lc{k} = permute(pagemtimes(M, 'transpose', T, 'none'), [3 1 2]);
end

% site d: nothing to sketch, the mode is left free
Lp = permute(Lc{d-1}, [2 3 1]);
Vm = reshape(Vcr(d), [vr(d), vn(d)]);
T  = reshape(pagemtimes(Lp, Vm), [Ar(d)*vn(d), 1, s]);
Yd = reshape(pagemtimes(Gd, 'transpose', T, 'none'), [An(d), s]).';

%% ---- second sweep: absorb the orthonormal factor into the core ----------
Q  = cell(1,d);
[Q{d}, ~] = qr(Yd.', 0);                         % [An(d), s]

Rc = cell(1,d);
T  = reshape(Gd * Q{d}, [Ar(d), Am(d), s]);
Rd = tensorprod(T, reshape(Vcr(d), [vr(d), vn(d)]), 2, 2);
Rd = reshape(Rd, [Ar(d), s, vr(d)]);
Rc{d} = permute(Rd, [1 3 2]);

for k = d-1:-1:2
    % Y^{(k)}: merge the two environments, then traverse the core once
    P  = tensorprod(Lc{k-1}, Vcr(k), 3, 1);
    P  = reshape(P, [s, Ar(k), vn(k), vr(k+1)]);
    Qq = tensorprod(P, Rc{k+1}, 4, 2);
    Qq = reshape(Qq, [s, Ar(k), vn(k), Ar(k+1), s]);
    Qq = reshape(permute(Qq, [2 3 4 1 5]), [Ar(k)*vn(k)*Ar(k+1), s*s]);
    LR = permute(reshape(Ak{k}*Qq, [An(k), s, s]), [1 3 2]);
    [Qk, ~] = qr(reshape(LR, [An(k)*s, s]), 0);
    Qk = reshape(Qk, [An(k), s, s]);

    % S^{(k)}: absorb Qk into the core, the transpose of the same matrix
    Gh = reshape(Ak{k}' * reshape(Qk, [An(k), s*s]), ...
                 [Ar(k), Am(k), Ar(k+1), s, s]);
    Tt = tensorprod(Vcr(k), Rc{k+1}, 3, 2);
    Tt = reshape(Tt, [vr(k), vn(k), Ar(k+1), s]);
    Rc{k} = tensorprod(Gh, Tt, [2 3 4], [2 3 4]);
    Rc{k} = permute(reshape(Rc{k}, [Ar(k), s, vr(k)]), [1 3 2]);

    Q{k} = permute(Qk, [3 1 2]);
end

% FIRST SITE.  The core of v meets the right environment before the operator
% core is touched; contracting the operator first would leave both of its
% mode indices free and cost a factor r^v more, in time and in memory.
W  = tensorprod(reshape(Vcr(1), [vn(1), vr(2)]), Rc{2}, 2, 2);  % [vn(1), Ar(2), s]
W  = reshape(W, [vn(1)*Ar(2), s]);
Q{1} = G1 * W;
Q{d} = Q{d}.';

w = tt_tensor;
w.d = d;
w.n = An(:);
w.r = [1; ones(d-1,1)*s; 1];
w.ps = cumsum([1; w.n.*w.r(1:d).*w.r(2:d+1)]);
Qvec = cell(1,d);
for k = 1:d, Qvec{k} = Q{k}(:); end
w.core = vertcat(Qvec{:});
end
function w = src_matvec_standard(A, v, s, varargin)
%SRC_MATVEC_STANDARD  SRC with both sweeps on the raw cores.
%
%   w = src_matvec_standard(A, v, s)
%   w = src_matvec_standard(A, v, s, Y)
%
%   Reference: Camano, Epperly and Tropp, arXiv:2504.06475.  This is SRC with
%   NOTHING pre-processed: the operator is never contracted with the random
%   matrices on its own, and every pass over an operator core carries the
%   rank r^v of the argument.  The second sweep is the shared-environment one
%   of src_matvec_standard, unchanged; only the first sweep
%   differs.
%
%   FIRST SWEEP.  At site k the left environment Lc{k-1}, of size
%   [s, r_A, r^v], meets the core of v and then the RAW operator core, and
%   only after that the column of Y{k} belonging to each sketch index:
%
%       T  = Lc{k-1} x V_k                     [r_A m,  r^v s]
%       R  = A_k x T      (sum over r_A, m)    [n,  r^v s,  r_A']   <- r_A^2 n m r^v s
%       Lc{k}(j,:,:) = sum_i Y{k}(i,j) R(i,:,j,:)
%
%   This is the fused order of the paper applied to SRC's first sweep; the
%   cores of A*v are never formed, which would cost r_A^2 n m (r^v)^2 and
%   would be strictly worse.  Compared with the pre-sketched first sweep of
%   src_matvec_standard, whose pass over the operator is
%   A_k' * Y{k} of weight r_A^2 n m s, this one costs a factor r^v more on
%   that pass: the total per interior site becomes about 2 r^v s instead of
%   r^v s + s, in units of r_A^2 n m.
%
%   Only the first s columns of Y are used; SRC has no oversampling.  Y{d} is
%   not used at all: the last site is left free for the orthonormalisation.

d  = length(v.n);
Am = A.m;  An = A.n;  Ar = A.r;  Aps = A.ps;  Acore = A.core;
vps = v.ps;  vr = v.r;  vn = v.n;  vcore = v.core;

Vcr = @(k) reshape(vcore(vps(k):vps(k+1)-1), [vr(k), vn(k), vr(k+1)]);
Acr = @(k) reshape(Acore(Aps(k):Aps(k+1)-1), [Ar(k), An(k), Am(k), Ar(k+1)]);

if nargin >= 4 && ~isempty(varargin{1})
    Y = varargin{1};
else
    Y = cell(1,d);
    for k = 1:d, Y{k} = randn(An(k), s); end
end
Ys = cell(1,d);
for k = 1:d, Ys{k} = Y{k}(:,1:s); end

%% first sweep, on the raw operator cores
Lc = cell(1,d);                                  % Lc{k} is [s, Ar(k+1), vr(k+1)]

% site 1, Ar(1) = vr(1) = 1: the operator core against the core of v, then Y
G1 = reshape(Acr(1), [An(1), Am(1), Ar(2)]);
T  = tensorprod(G1, reshape(Vcr(1), [vn(1), vr(2)]), 2, 1);    % [An(1), Ar(2), vr(2)]
T  = reshape(T, [An(1), Ar(2)*vr(2)]);
Lc{1} = reshape(Ys{1}.' * T, [s, Ar(2), vr(2)]);

for k = 2:d-1
    % [An, Ar*Am, Ar']: the two contracted indices adjacent, Ar fastest
    Ak = reshape(permute(Acr(k), [2 1 3 4]), [An(k), Ar(k)*Am(k), Ar(k+1)]);
    Lp = permute(Lc{k-1}, [2 3 1]);                            % [Ar, vr, s]
    Vm = reshape(Vcr(k), [vr(k), vn(k)*vr(k+1)]);
    T  = pagemtimes(Lp, Vm);                                   % [Ar, vn*vr', s]
    T  = reshape(T, [Ar(k)*vn(k), vr(k+1)*s]);                 % rows (Ar, vn=Am)
    R  = pagemtimes(Ak, T);                                    % [An, vr'*s, Ar']
    R  = reshape(R, [An(k), vr(k+1), s, Ar(k+1)]);
    R  = sum(R .* reshape(Ys{k}, [An(k), 1, s]), 1);           % Khatri-Rao: column j of Y{k}
    Lc{k} = permute(reshape(R, [vr(k+1), s, Ar(k+1)]), [2 3 1]);   % [s, Ar', vr']
end

% site d: nothing to sketch, the mode is left free
Lp = permute(Lc{d-1}, [2 3 1]);                  % [Ar(d), vr(d), s]
Vm = reshape(Vcr(d), [vr(d), vn(d)]);
T  = reshape(pagemtimes(Lp, Vm), [Ar(d)*vn(d), 1, s]);
G  = reshape(permute(Acr(d), [1 3 2 4]), [Ar(d)*Am(d), An(d)]);
Yd = reshape(pagemtimes(G, 'transpose', T, 'none'), [An(d), s]).';   % [s, An(d)]

%% second sweep: QR at each site, environment shared between Y and S
% identical to src_matvec_standard
Q  = cell(1,d);
[Q{d}, ~] = qr(Yd.', 0);                         % [An(d), s]

Rc = cell(1,d);
T  = tensorprod(Acr(d), Q{d}, 2, 1);
T  = reshape(T, [Ar(d), Am(d), s]);
Rd = tensorprod(T, reshape(Vcr(d), [vr(d), vn(d)]), 2, 2);
Rd = reshape(Rd, [Ar(d), s, vr(d)]);
Rc{d} = permute(Rd, [1 3 2]);

for k = d-1:-1:2
    W = tensorprod(Vcr(k), Rc{k+1}, 3, 2);
    W = reshape(W, [vr(k), vn(k), Ar(k+1), s]);
    U = tensorprod(Acr(k), W, [3 4], [2 3]);     % [Ar(k), An(k), vr(k), s]
    U = reshape(U, [Ar(k), An(k), vr(k), s]);
    LR = tensorprod(U, Lc{k-1}, [1 3], [2 3]);   % [An(k), s, s]
    LR = reshape(LR, [An(k)*s, s]);
    [Qk, ~] = qr(LR, 0);
    Qk = reshape(Qk, [An(k), s, s]);
    Rc{k} = reshape(tensorprod(U, Qk, [2 4], [1 2]), [Ar(k), vr(k), s]);
    Q{k}  = permute(Qk, [3 1 2]);                % [s, An(k), s]
end

% first site, v against the environment before the operator (see the
% comment in src_matvec_standard)
W  = tensorprod(reshape(Vcr(1), [vn(1), vr(2)]), Rc{2}, 2, 2);  % [vn(1), Ar(2), s]
W  = reshape(W, [vn(1)*Ar(2), s]);
Q{1} = reshape(Acr(1), [An(1), Am(1)*Ar(2)]) * W;
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
function w = src_recover_sum(A, V, s, Yd_all, Lc_all, maxbytes)
%SRC_RECOVER_SUM  SRC's second sweep for a sum of arguments.
%
%   w = src_recover_sum(A, V, s, Yd_all, Lc_all)
%   w = src_recover_sum(A, V, s, Yd_all, Lc_all, maxbytes)
%
%   V is a cell array of TT tensors and Yd_all{i}, Lc_all{i} are the outputs
%   of src_sketch on V{i}.  Returns a TT tensor of rank s
%   approximating
%
%       A*V{1} + A*V{2} + ... + A*V{N}.
%
%   HOW THE SUM IS HANDLED.  Everything that the QR sees is summed over the
%   arguments first, so ONE orthonormal factor is computed per site and shared:
%
%       Q_d = orth( sum_i Yd_i' ),        Q_k = orth( sum_i LR_k^{(i)} ),
%
%   while the right environments Rc^{(i)} are kept per argument, because they
%   are what carries each argument's own information forward.  The site-1 core
%   is the sum of the contributions.
%
%   WHY THIS NEEDS THE ARGUMENTS AGAIN.  Q_k is built from Q_{k+1} and from
%   every V{i}, so site k cannot be reached without revisiting the whole
%   collection: SRC needs d-1 passes over the arguments, or, as here, the
%   arguments kept in memory.  The sketches of STTA are a fixed linear map and
%   need one pass and no storage.  That is the difference the streaming
%   experiment measures.
%
%   WHAT IS OPTIMISED HERE, against a direct transcription of the algorithm:
%
%     * the first site contracts the core of v against the right environment
%       BEFORE the operator, and the operator is applied ONCE to the sum of
%       those environments rather than once per argument.  The other order
%       leaves both mode indices of A_1 free and costs n m r^v s per argument;
%     * the environment U_k^{(i)} is formed with one matrix product on the
%       core as it is stored, [r n, m r'], with no transposition;
%     * U_k^{(i)} is cached while it fits in MAXBYTES (default 512 MB) and
%       recomputed otherwise, so the routine does not silently allocate
%       N r_A n r^v s numbers;
%     * the cores of V{i} are read when needed instead of being copied into a
%       table, which would double the memory the experiment is measuring.

if nargin < 6 || isempty(maxbytes), maxbytes = 512e6; end

N = numel(V);
d = length(V{1}.n);
Am = A.m;  An = A.n;  Ar = A.r;  Aps = A.ps;  Acore = A.core;
Acr = @(k) reshape(Acore(Aps(k):Aps(k+1)-1), [Ar(k), An(k), Am(k), Ar(k+1)]);

% core of V{i} at site k, read when needed: no table of copies
Vcr   = @(i,k) reshape(V{i}.core(V{i}.ps(k):V{i}.ps(k+1)-1), ...
                       [V{i}.r(k), V{i}.n(k), V{i}.r(k+1)]);
vrank = @(i,k) double(V{i}.r(k));

Q = cell(1,d);

%% ---- last site: one QR for the summed sketch ------------------------------
Yd = Yd_all{1};
for i = 2:N, Yd = Yd + Yd_all{i}; end
[Q{d}, ~] = qr(Yd.', 0);                                   % [An(d), s]

% the core of A at site d, contracted with Q{d} once: it does not depend on i
Gd = reshape(permute(reshape(Acore(Aps(d):Aps(d+1)-1), [Ar(d), An(d), Am(d)]), ...
                     [2 1 3]), [An(d), Ar(d)*Am(d)]);
T  = reshape(Gd.' * Q{d}, [Ar(d), Am(d), s]);

Rcur = cell(1,N);                    % the right environment of every argument
for i = 1:N
    Rd = tensorprod(T, reshape(Vcr(i,d), [vrank(i,d), V{i}.n(d)]), 2, 2);
    Rd = reshape(Rd, [Ar(d), s, vrank(i,d)]);
    Rcur{i} = permute(Rd, [1 3 2]);                        % [Ar(d), vr(d), s]
end

%% ---- interior sites: one QR each, environments kept per argument ---------
for k = d-1:-1:2
    % the core as it is stored: rows (a, i), columns (m, a'), a plain reshape
    Aright = reshape(Acr(k), [Ar(k)*An(k), Am(k)*Ar(k+1)]);

    vmax   = max(cellfun(@(x) double(x.r(k)), V));
    keepU  = N*8*Ar(k)*An(k)*vmax*s <= maxbytes;
    Ucache = cell(1, N*keepU);

    LR = zeros(An(k)*s, s);
    for i = 1:N
        U = src_environment(Aright, Vcr(i,k), Rcur{i}, Ar(k), An(k), s, vrank(i,k));
        if keepU, Ucache{i} = U; end
        LR = LR + reshape(tensorprod(U, Lc_all{i}{k-1}, [1 3], [2 3]), [An(k)*s, s]);
    end

    [Qk, ~] = qr(LR, 0);
    Qk = reshape(Qk, [An(k), s, s]);

    for i = 1:N
        if keepU
            U = Ucache{i};
        else
            U = src_environment(Aright, Vcr(i,k), Rcur{i}, Ar(k), An(k), s, vrank(i,k));
        end
        Rcur{i} = reshape(tensorprod(U, Qk, [2 4], [1 2]), [Ar(k), vrank(i,k), s]);
    end
    clear Ucache

    Q{k} = permute(Qk, [3 1 2]);                           % [s, An(k), s]
end

%% ---- first site: the operator is applied once, to the summed environment -
% Contracting A_1 with the environment first would leave both of its mode
% indices free -- n m r^v s numbers per argument -- to produce an n x s
% matrix.  Here the core of v meets the environment first, and since A_1 is
% the same for every argument the sum is taken before the operator is applied.
W = [];
for i = 1:N
    Wi = tensorprod(reshape(Vcr(i,1), [V{i}.n(1), vrank(i,2)]), Rcur{i}, 2, 2);
    Wi = reshape(Wi, [V{i}.n(1)*Ar(2), s]);
    if isempty(W), W = Wi; else, W = W + Wi; end
end
Q{1} = reshape(Acr(1), [An(1), Am(1)*Ar(2)]) * W;
Q{d} = Q{d}.';

%% ---- assemble -------------------------------------------------------------
w = tt_tensor;
w.d = d;
w.n = An(:);
w.r = [1; ones(d-1,1)*s; 1];
w.ps = cumsum([1; w.n.*w.r(1:d).*w.r(2:d+1)]);
Qvec = cell(1,d);
for k = 1:d, Qvec{k} = Q{k}(:); end
w.core = vertcat(Qvec{:});
end

%% ==========================================================================
function U = src_environment(Aright, Vk, Rk1, Ark, Ank, s, vrk)
%ENVIRONMENT  U(a, i, b, j) = sum A_k(a,i,m,a') V_k(b,m,b') R_{k+1}(a',b',j).
%   One matrix product on the core as it is stored; the permutation falls on
%   the small object, not on the operator.
    W = tensorprod(Vk, Rk1, 3, 2);                      % [vr, vn, Ar', s]
    W = reshape(W, [vrk, size(Vk,2), size(Rk1,1), s]);
    W = reshape(permute(W, [2 3 1 4]), [size(Vk,2)*size(Rk1,1), vrk*s]);
    U = reshape(Aright * W, [Ark, Ank, vrk, s]);
end
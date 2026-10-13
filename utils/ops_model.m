function C = ops_model(d, m, n, rA, rv, s, p)
%OPS_MODEL  Operation counts, one term per contraction as the code writes it.
%
%   C = ops_model(d, m, n, rA, rv, s, p)
%
%   Returns a struct with a total per method and the breakdown that explains
%   it.  Counts are multiply-adds, (output size) times (contracted size), and
%   are summed over the ACTUAL sites: the first and the last are not interior
%   and do not cost the same.
%
%   WHY THIS REPLACES cost_model.  That routine multiplies a per-core weight
%   by d.  Asymptotically in d it is right, but only d-2 sites are interior,
%   so at d = 4 it overstates by roughly two -- and not equally for every
%   method, because the boundary sites differ enormously between them.  Read
%   against measurements it produced a spurious "efficiency gap": ours and SRC
%   appeared to run at 30 and 35 operations per second, when in truth both run
%   near 15 and the ratio of times already follows the ratio of operations.
%
%   THE TWO TERMS WORTH READING.
%
%   C.ours_Apass is the product against the operator core in the assembly, of
%   weight r_A^2 n^2 s ell.  It does not depend on r^v.  Everything else we
%   pay is quadratic in r^v and is paid by every method alike, so the fraction
%   C.ours_Apass / C.ours is how much of our cost the comparison is even
%   about: 91 per cent at r^v = 16, 59 at r^v = 60, 24 at r^v = 140 for
%   m = n = 512, s = 8.  That fraction falling is the regime condition
%   s m r^A >~ (r^v)^2 being lost, and it is why every advantage in this paper
%   has a maximum in r^v rather than growing.
%
%   C.src_site1 is SRC's first site, contracted in the order that keeps it
%   cheap: the core of v against the right environment first, then the
%   operator once.  C.src_site1_naive is the order originally coded --
%       T = tensorprod(Acr(1), Rc{2}, 4, 1);
%   -- which leaves both mode indices of the operator free and so costs
%   n^2 r^v s r_A instead of about n^2 r^A s, a factor r^v, while allocating
%   n m r^v s numbers instead of n r^A s.  That allocation, 2.68 GB at
%   n = 1024, r^v = 40, s = 8, is what stopped the comparison at m = 1024.
%   C.src_shared_naive and C.src_absorb_naive are the totals with it.

l  = s + p;
Ar = [1, rA*ones(1,d-1), 1];
vr = [1, rv*ones(1,d-1), 1];
A  = @(k) Ar(k);    Ap = @(k) Ar(k+1);      % r_A(k), r_A(k+1)
V  = @(k) vr(k);    Vp = @(k) vr(k+1);

%% ---- the assembly, shared by the sequential, fused and pre-sketched routes
asm = 0;
for k = 1:d-1, asm = asm + l*s*Ap(k)*Vp(k); end                  % Omega
asm = asm + n*Ap(1)*s*Vp(1) + n*s*n*Ap(1);                       % Psi{1}
Apass = 0;
for k = 2:d-1
    asm   = asm + V(k)*n*Ap(k)*s*Vp(k);                          % W0
    asm   = asm + A(k)*l*n*Ap(k)*s*V(k);                         % Wt
    Apass = Apass + n*s*l*A(k)*n*Ap(k);                          % Ak*Wt
end
asm = asm + Apass + A(d)*l*n*V(d) + l*n*A(d)*n;                  % Psi{d}

%% ---- the two pre-sketched sweeps ----------------------------------------
sw = A(d)*s*V(d)*n;
for k = d:-1:2
    sw = sw + V(k-1)*n*A(k)*s*V(k) + A(k-1)*V(k-1)*s*A(k)*m;
end
sw = sw + Ap(1)*l*Vp(1)*n;
for k = 1:d-1
    sw = sw + n*Vp(k+1)*Ap(k)*l*Vp(k) + Ap(k+1)*Vp(k+1)*l*Ap(k)*m;
end

%% ---- the fused order, with the environment shared with the assembly -----
% Both sweeps traverse the raw cores.  The right sweep builds
%     U_k(a,i,b,j) = sum A_k(a,i,m,a') V_k(b,m,b') R_{k+1}(a',b',j),
% the object SRC calls the shared environment, and uses it TWICE: contracted
% with the column of Y it gives R_k, contracted with the left carry it gives
% Psi_k.  So the assembly costs no pass over A at all and the m*n weight is
% r^v(2s+ell) rather than r^v(2s+ell) + s(s+ell).
fs = Ap(1)*n*Vp(1)*m + Ap(1)*Vp(1)*l*n;                          % left sweep, site 1
for k = 2:d-1
    fs = fs + n*Vp(k)*A(k)*l*V(k);                               % W = V_k x Lc
    fs = fs + Ap(k)*n*Vp(k)*l*A(k)*m;                            % the pass over A_k
    fs = fs + Ap(k)*Vp(k)*l*n;                                   % contraction with Y_k
end
fs = fs + A(d)*n*V(d)*m + A(d)*V(d)*s*n + l*n*A(d)*V(d);         % right sweep, site d
for k = d-1:-1:2
    fs = fs + V(k)*n*Ap(k)*s*Vp(k);                              % W = V_k x Rc
    fs = fs + A(k)*n*V(k)*s*m*Ap(k);                             % U_k: the pass over A_k
    fs = fs + A(k)*V(k)*s*n;                                     % R_k  = Y_k x U_k
    fs = fs + l*n*s*A(k)*V(k);                                   % Psi_k = L_{k-1} x U_k
end
fs = fs + n*Ap(1)*s*Vp(1) + n*s*m*Ap(1);                         % Psi{1}
for k = 1:d-1, fs = fs + l*s*Ap(k)*Vp(k); end                    % Omega

%% ---- SRC, first sweep, pre-sketched (SRC_..._absorb) --------------------
f1 = Ap(1)*s*Vp(1)*n;
for k = 2:d-1
    f1 = f1 + A(k)*n*Vp(k)*s*V(k) + Ap(k)*Vp(k)*s*A(k)*m;
end
f1 = f1 + A(d)*n*s*V(d) + n*s*A(d)*n;

%% ---- SRC, first sweep on the raw cores (SRC_..._raw) --------------------
% the operator core meets the environment and the core of v, NOT the random
% matrix alone: its pass carries r^v, weight r_A^2 n m r^v s
f1raw = n*m*Ap(1)*Vp(1) + s*n*Ap(1)*Vp(1);
for k = 2:d-1
    f1raw = f1raw + A(k)*m*Vp(k)*s*V(k);                         % Lc x V_k
    f1raw = f1raw + n*A(k)*m*Vp(k)*s*Ap(k);                      % A_k x T
    f1raw = f1raw + n*Vp(k)*s*Ap(k);                             % Khatri-Rao with Y
end
f1raw = f1raw + A(d)*n*s*V(d) + n*s*A(d)*n;

%% ---- SRC, second sweep, shared environments -----------------------------
sh = A(d)*n*s*n + A(d)*s*V(d)*n;
for k = d-1:-1:2
    sh = sh + V(k)*n*Ap(k)*s*Vp(k);                              % W
    sh = sh + A(k)*n*V(k)*s*n*Ap(k);                             % U: r_A^2 n^2 r^v s
    sh = sh + n*s*s*A(k)*V(k) + n*s*s*s;                         % LR and its QR
    sh = sh + A(k)*V(k)*s*n*s;                                   % Rc
end

%% ---- SRC, second sweep, orthonormal factor absorbed ---------------------
ab = A(d)*n*s*n + A(d)*s*V(d)*n;
for k = d-1:-1:2
    ab = ab + s*A(k)*n*Vp(k)*V(k);                               % P
    ab = ab + s*A(k)*n*Ap(k)*s*Vp(k);                            % Qq
    ab = ab + n*s*s*A(k)*n*Ap(k);                                % Ak*Qq: r_A^2 n^2 s^2
    ab = ab + n*s*s*s;                                           % QR
    ab = ab + A(k)*n*Ap(k)*s*s*n;                                % Ak'*Qk: again
    ab = ab + V(k)*n*Ap(k)*s*Vp(k);                              % Tt
    ab = ab + A(k)*s*V(k)*n*Ap(k)*s;                             % Rc
end

%% ---- SRC, the first site -------------------------------------------------
% Contracting the core of v with the right environment first, which is what
% the reordered implementation does.  The order originally coded there
% contracts the operator core with the environment instead, leaving both mode
% indices free: n^2 r^v s r_A operations and an intermediate of n^2 r^v s
% numbers, 2.68 GB at n = 1024, r^v = 40, s = 8.  Both are reported.
b1      = n*Ap(1)*s*Vp(1) + n*s*n*Ap(1);
b1naive = n*n*Vp(1)*s*Ap(1) + n*s*n*Vp(1);

C.ours       = sw + asm;
C.fused      = fs;                 % the assembly is inside fs
C.src_shared = f1 + sh + b1;
C.src_absorb = f1 + ab + b1;
C.src_raw    = f1raw + sh + b1;         % SRC_..._raw: raw first sweep, shared second

C.assembly    = asm;
C.ours_sweeps = sw;
C.ours_Apass  = Apass;                  % the r_A^2 n^2 s ell product
C.src_sweep1  = f1;
C.src_sweep1_raw = f1raw;
C.src_sweep2_shared = sh;
C.src_sweep2_absorb = ab;
C.src_site1       = b1;
C.src_site1_naive = b1naive;
C.src_shared_naive = C.src_shared - b1 + b1naive;
C.src_absorb_naive = C.src_absorb - b1 + b1naive;

C.regime      = s*m*rA/rv^2;            % >~ 1 is the favourable regime
C.site1_bytes = 8*n*m*rv*s;             % what SRC's first site allocates
end
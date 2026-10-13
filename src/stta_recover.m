function w = stta_recover(Omega, Psi)
%STTA_RECOVER  Recover a TT tensor from its STTA sketches.
%
%   w = stta_recover(Omega, Psi)
%
%   Omega{k}, k = 1..d-1, of size (s+l) x s
%   Psi{1}                of size n(1) x s
%   Psi{k}, k = 2..d-1,   of size (s+l) x (n(k)*s)
%   Psi{d}                of size (s+l) x n(d)
%
%   The cores are formed as  C{1} = Psi{1} V1 S1^-1,
%   C{k} = U_{k-1}' Psi{k} Vk Sk^-1,  C{d} = U_{d-1}' Psi{d},
%   which is the product form of Omega_{k-1}^+ Psi_k with the right factor of
%   each pseudoinverse pushed into the preceding core.  Singular values below
%   TOL times the largest are discarded.

TOL = 5*eps;

d = numel(Psi);
s = size(Omega{1}, 2);

n = zeros(1,d);
n(1) = size(Psi{1}, 1);
for k = 2:d-1
    n(k) = size(Psi{k}, 2)/s;
end
n(d) = size(Psi{d}, 2);

C = cell(1,d);

[Uold, S, V] = svd(Omega{1}, 'econ');
dS = diag(S);  nz = dS > TOL*dS(1);
iS = zeros(size(dS));  iS(nz) = 1./dS(nz);
C{1} = (Psi{1}*V) .* iS.';

for k = 2:d-1
    [U, S, V] = svd(Omega{k}, 'econ');
    dS = diag(S);  nz = dS > TOL*dS(1);
    iS = zeros(size(dS));  iS(nz) = 1./dS(nz);
    M    = reshape(Uold.'*Psi{k}, [s*n(k), s]);
    C{k} = (M*V) .* iS.';
    Uold = U;
end

C{d} = Uold.'*Psi{d};

w = tt_tensor;
w.d = d;
w.n = n(:);
w.r = [1; ones(d-1,1)*s; 1];
w.ps = cumsum([1; w.n.*w.r(1:d).*w.r(2:d+1)]);
Cvec = cell(1,d);
for k = 1:d, Cvec{k} = C{k}(:); end
w.core = vertcat(Cvec{:});
end

function [W, H] = opnmf_adv(X, K, w0, initMeth, max_iter, tol, iter0, save_step, outputdir)
%% OPNMF_adv
%
% Memory-aware OPNMF with optional NaN handling.
%
% -------------------------------------------------------------------------
% Dimensions
% -------------------------------------------------------------------------
% D = number of features / rows / voxels / vertices / variables
% N = number of samples  / columns / subjects / images / observations
% K = number of components / latent nonnegative bases
%
% X is a D-by-N matrix.
% W is a D-by-K basis matrix.
% H is a K-by-N coefficient matrix.
%
% -------------------------------------------------------------------------
% External interface
% -------------------------------------------------------------------------
% [W, H] = opnmf_adv(X, K, w0, initMeth, max_iter, tol, iter0, save_step, outputdir)
%
% This function keeps the original brainparts OPNMF interface as much as
% possible, while adding a practical missing-data branch for NaN entries.
%
% -------------------------------------------------------------------------
% Core computational schemes
% -------------------------------------------------------------------------
% Scheme A: complete-data branch (exact original OPNMF)
% -------------------------------------------------------------------------
% If X contains no NaN values, this function executes the exact original
% OPNMF multiplicative update, evaluated in the memory-aware order used by
% opnmf_mem.m:
%
%   W <- W .* (X * (X' * W)) ./ (W * ((W' * X) * (X' * W)))
%
% This is algebraically equivalent to the XX = X*X' implementation but
% avoids explicitly forming a D-by-D Gram matrix.
%
% Scheme B: missing-data branch (low-memory masked projective extension)
% -------------------------------------------------------------------------
% If X contains NaN values, the exact original update is not directly
% applicable. We encode the data as:
%
%   X0 = X with NaN entries replaced by 0
%   M  = logical observation mask, M = ~isnan(X)
%
% Then we alternate:
%
%   (1) Weighted projective coefficient update
%       H(:,n) = (W' * x0_n) ./ ((W.^2)' * m_n)
%
%   (2) Masked multiplicative W update
%       W <- W .* (X0 * H') ./ ((M .* (W * H)) * H')
%
% evaluated in sample blocks so that no D-by-D matrix is formed.
%
% Important note on Scheme B
% -------------------------------------------------------------------------
% The missing-data branch is a practical low-memory extension that keeps
% the projective spirit of OPNMF. It is exact with respect to the original
% brainparts implementation only in the complete-data case. For incomplete
% data it should be viewed as a masked approximation designed to avoid the
% D-by-D memory bottleneck.
%
% -------------------------------------------------------------------------
% Inputs
% -------------------------------------------------------------------------
% X          : D-by-N nonnegative data matrix.
%              Missing entries, if any, must be encoded as NaN.
% K          : number of components, 1 <= K <= min(size(X)).
% w0         : optional initialization of W.
%              It may be given in either:
%              (a) original row space   : D_original-by-K
%              (b) reduced row space    : D_reduced-by-K
%              where "reduced" means after background-row removal.
% initMeth   : initialization method if w0 is empty
%              0 = random
%              1 = NNDSVD      (default, same as original code)
%              2 = NNDSVDa
%              3 = NNDSVDar
%              4 = NNDSVD with randomized SVD backend when available
% max_iter   : maximum number of iterations (default: 50000)
% tol        : convergence tolerance on relative Frobenius change of W
%              (default: 1e-5)
% iter0      : starting iteration index, useful for restarting (default: 1)
% save_step  : save intermediate reduced-space W every save_step iterations
%              (default follows original code: floor(max_iter/10), but is
%              clamped to at least 1 to avoid a zero-step issue)
% outputdir  : optional output directory for intermediate results
%
% -------------------------------------------------------------------------
% Outputs
% -------------------------------------------------------------------------
% W          : D_original-by-K basis matrix in the original row space
% H          : K-by-N coefficient matrix
%
% -------------------------------------------------------------------------
% References
% -------------------------------------------------------------------------
% Yang, Z. and Oja, E. (2010).
% Linear and nonlinear projective nonnegative matrix factorization.
% IEEE Transactions on Neural Networks, 21(5), 734-749.
%
% Sotiras, A., Resnick, S. M., and Davatzikos, C. (2015).
% Finding imaging patterns of structural covariance via Non-Negative
% Matrix Factorization. NeuroImage, 108, 1-16.
%
% Boutsidis, C. and Gallopoulos, E. (2008).
% SVD-based initialization: A head start for nonnegative matrix
% factorization. Pattern Recognition, 41(4), 1350-1362.
%
% Halko, N., Martinsson, P.-G., and Tropp, J. A. (2011).
% Finding structure with randomness: Probabilistic algorithms for
% constructing approximate matrix decompositions. SIAM Review, 53(2),
% 217-288.

tiny = 1e-16;
check_step = 100;

% -------------------------------------------------------------------------
% Basic validation
% -------------------------------------------------------------------------
[Dinit, N] = size(X);

if ~isscalar(K) || ~isnumeric(K) || ~isfinite(K) || K < 1 || K > min(Dinit, N) || K ~= round(K)
    error('opnmf_nan:badK', ...
        'K must be a positive integer no larger than min(size(X)).');
end

if ~ismatrix(X) || ~isnumeric(X) || ~isreal(X)
    error('opnmf_nan:badX', 'X must be a real numeric matrix.');
end

obs = X(~isnan(X));
if any(obs < 0)
    error('opnmf_nan:badX', ...
        'Observed entries of X must be nonnegative. Missing entries must be NaN.');
end
if any(~isfinite(obs))
    error('opnmf_nan:badX', ...
        'Observed entries of X must be finite. Use NaN only for missing values.');
end

if nargin < 3 || isempty(w0)
    w0 = [];
end

if nargin < 4 || isempty(initMeth)
    initMeth = 1;
end
if ~isscalar(initMeth) || ~isnumeric(initMeth) || ~isfinite(initMeth)
    error('opnmf_nan:badInit', 'initMeth must be a finite scalar.');
end

if nargin < 5 || isempty(max_iter)
    max_iter = 50000;
end
if ~isscalar(max_iter) || ~isnumeric(max_iter) || ~isfinite(max_iter) || max_iter < 1 || max_iter ~= round(max_iter)
    error('opnmf_nan:badMaxIter', 'max_iter must be a positive integer.');
end

if nargin < 6 || isempty(tol)
    tol = 1e-5;
end
if ~isscalar(tol) || ~isnumeric(tol) || ~isfinite(tol) || tol < 0
    error('opnmf_nan:badTol', 'tol must be a finite nonnegative scalar.');
end

if nargin < 7 || isempty(iter0)
    iter0 = 1;
end
if ~isscalar(iter0) || ~isnumeric(iter0) || ~isfinite(iter0) || iter0 < 1 || iter0 ~= round(iter0)
    error('opnmf_nan:badIter0', 'iter0 must be a positive integer.');
end
if iter0 > max_iter
    error('opnmf_nan:badIter0', 'iter0 must not exceed max_iter.');
end

if nargin < 8 || isempty(save_step)
    save_step = floor(max_iter / 10);   % original default
end

if nargin < 9 || isempty(outputdir)
    outputdir = '';
end

if ~isscalar(save_step) || ~isnumeric(save_step) || ~isfinite(save_step) || save_step < 0
    error('opnmf_nan:badSaveStep', 'save_step must be a nonnegative scalar.');
end
save_step = max(1, round(save_step));

hasOutputDir = false;
if nargin >= 9 && ~isempty(outputdir)
    if ~ischar(outputdir)
        error('opnmf_nan:badDir', 'outputdir must be a character vector.');
    end
    hasOutputDir = true;
    if ~exist(outputdir, 'dir')
        success = mkdir(outputdir);
        if ~success
            error('opnmf_nan:badDir', ...
                'Output directory "%s" could not be created.', outputdir);
        end
    end
end

hasMissing = any(isnan(X(:)));

% -------------------------------------------------------------------------
% Remove background rows
% -------------------------------------------------------------------------
% Original code removes rows whose stackwise mean is zero.
% Here:
% - if no NaN exists, we do exactly the original rule
% - if NaN exists, we compute row means only over observed entries
%   and also remove rows with no observed entries

if hasMissing
    Mfull = ~isnan(X);
    Xsum = X;
    Xsum(~Mfull) = 0;

    row_count = sum(Mfull, 2);
    row_mean  = sum(Xsum, 2) ./ max(row_count, 1);
    keep_rows = (row_count > 0) & (row_mean > 0);
else
    keep_rows = (mean(X, 2) > 0);
end

X = X(keep_rows, :);
D = size(X, 1);

if D == 0
    error('opnmf_nan:emptyData', ...
        'No informative rows remain after background/all-missing row removal.');
end

if K > D
    error('opnmf_nan:badKReduced', ...
        ['After background removal, the remaining number of rows is ' ...
         'smaller than K.']);
end

% -------------------------------------------------------------------------
% Initialize W
% -------------------------------------------------------------------------
W = initialize_W_local(X, K, w0, initMeth, keep_rows, Dinit, hasMissing, tiny);

% -------------------------------------------------------------------------
% Optimization
% -------------------------------------------------------------------------
if hasMissing
    [W, H] = run_masked_lowmem_local( ...
        X, W, max_iter, tol, iter0, save_step, outputdir, hasOutputDir, check_step);
else
    [W, H] = run_exact_original_mem_local( ...
        X, W, max_iter, tol, iter0, save_step, outputdir, hasOutputDir, check_step);
end

% -------------------------------------------------------------------------
% Reorder components
% -------------------------------------------------------------------------
[W, H] = reorder_components_local(W, H);

% -------------------------------------------------------------------------
% Restore W to the original row space
% -------------------------------------------------------------------------
Wfull = zeros(Dinit, K);
Wfull(keep_rows, :) = W;
W = Wfull;

end

% =========================================================================
function W = initialize_W_local(X, K, w0, initMeth, keep_rows, Dinit, hasMissing, tiny)

D = size(X, 1);

if nargin < 3 || isempty(w0)
    switch initMeth
        case 0
            W = rand(D, K);

        case 1
            if hasMissing
                [W, ~] = NNDSVD_nanaware_local(X, K, 0);
            else
                [W, ~] = NNDSVD_local(X, K, 0);
            end

        case 2
            if hasMissing
                [W, ~] = NNDSVD_nanaware_local(X, K, 1);
            else
                [W, ~] = NNDSVD_local(X, K, 1);
            end

        case 3
            if hasMissing
                [W, ~] = NNDSVD_nanaware_local(X, K, 2);
            else
                [W, ~] = NNDSVD_local(X, K, 2);
            end

        case 4
            if hasMissing
                [W, ~] = NNDSVD_nanaware_local(X, K, 3);
            else
                [W, ~] = NNDSVD_local(X, K, 3);
            end

        otherwise
            if hasMissing
                [W, ~] = NNDSVD_nanaware_local(X, K, 0);
            else
                [W, ~] = NNDSVD_local(X, K, 0);
            end
    end
else
    W = w0;

    % Accept either original-space or reduced-space initialization.
    if size(W, 1) == Dinit && size(W, 2) == K
        W = W(keep_rows, :);
    end

    if size(W, 1) ~= D || size(W, 2) ~= K || any(W(:) < 0) || any(~isfinite(W(:)))
        error('opnmf_nan:badW0', ...
            ['w0 must be a finite nonnegative matrix of size ' ...
             'D_reduced-by-K or D_original-by-K.']);
    end
end

W = max(W, tiny);

end

% =========================================================================
function [W, H] = run_exact_original_mem_local(X, W, max_iter, tol, iter0, save_step, outputdir, hasOutputDir, check_step)
%% Exact original OPNMF branch for complete data
%
% This is the exact multiplicative update used by the original memory-aware
% implementation opnmf_mem.m:
%
%   W <- W .* (X * (X' * W)) ./ (W * ((W' * X) * (X' * W)))

tiny = 1e-16;
[D, K] = size(W);

for iter = iter0:max_iter
    W_old = W;

    % if mod(iter, check_step) == 0
    %     fprintf('iter=% 5d ', iter);
    % end

    XtW = X' * W;                    % N-by-K
    NumerW = X * XtW;                % D-by-K
    DenomW = W * (XtW' * XtW);       % D-by-K
    DenomW = max(DenomW, tiny);

    W = W .* (NumerW ./ DenomW);

    % Match original thresholding behavior
    W(W < tiny) = tiny;

    nW = norm(W);
    if ~isfinite(nW) || nW <= 0
        error('opnmf_nan:numericalFailure', ...
            'Normalization failed in complete-data branch.');
    end
    W = W ./ nW;

    diffW = norm(W_old - W, 'fro') / max(norm(W_old, 'fro'), tiny);

    % if mod(iter, check_step) == 0
    %     fprintf('diffW=%0.6e\n', diffW);
    % end

    if diffW < tol
        break;
    end

    if hasOutputDir && mod(iter, save_step) == 0
        save_intermediate_local(outputdir, iter, D, K, W);
    end
end

H = W' * X;

end

% =========================================================================
function [W, H] = run_masked_lowmem_local(X, W, max_iter, tol, iter0, save_step, outputdir, hasOutputDir, check_step)
%% Low-memory masked projective branch for data containing NaN
%
% Missing-data encoding:
%   X0 = X with NaN replaced by 0
%   M  = logical observation mask
%
% Iteration:
%   1) H update using a weighted projective rule
%   2) W update using a masked multiplicative rule
%
% All large products are evaluated in column blocks to avoid forming
% D-by-D matrices.

tiny = 1e-16;

M = ~isnan(X);
X0 = X;
X0(~M) = 0;

[D, K] = size(W);
N = size(X0, 2);

blockN = choose_block_size_local(D, N);

for iter = iter0:max_iter
    W_old = W;

    if mod(iter, check_step) == 0
        fprintf('iter=% 5d ', iter);
    end

    % -------------------------------------------------------------
    % Step 1: masked projective coefficient update
    % -------------------------------------------------------------
    H = compute_H_masked_blockwise_local(W, X0, M, blockN);

    % -------------------------------------------------------------
    % Step 2: masked multiplicative W update
    % -------------------------------------------------------------
    [NumerW, DenomW] = compute_W_terms_masked_blockwise_local(W, H, X0, M, blockN);
    DenomW = max(DenomW, tiny);

    W = W .* (NumerW ./ DenomW);

    % Keep entries away from exact zero
    W(W < tiny) = tiny;

    nW = norm(W);
    if ~isfinite(nW) || nW <= 0
        error('opnmf_nan:numericalFailure', ...
            'Normalization failed in missing-data branch.');
    end
    W = W ./ nW;

    diffW = norm(W_old - W, 'fro') / max(norm(W_old, 'fro'), tiny);

    if mod(iter, check_step) == 0
        fprintf('diffW=%0.6e\n', diffW);
    end

    if diffW < tol
        break;
    end

    if hasOutputDir && mod(iter, save_step) == 0
        save_intermediate_local(outputdir, iter, D, K, W);
    end
end

% Recompute H once more for the final W
H = compute_H_masked_blockwise_local(W, X0, M, blockN);

end

% =========================================================================
function H = compute_H_masked_blockwise_local(W, X0, M, blockN)
%% Weighted projective coefficient update in column blocks
%
% For sample n:
%   H(:,n) = (W' * x0_n) ./ ((W.^2)' * m_n)
%
% This is a diagonal approximation to the masked normal equations.

tiny = 1e-16;
K = size(W, 2);
N = size(X0, 2);

H = zeros(K, N);
WW = W .^ 2;

for first = 1:blockN:N
    last = min(first + blockN - 1, N);
    idx = first:last;

    XB = X0(:, idx);
    MB = M(:, idx);

    NumerH = W' * XB;
    DenomH = WW' * double(MB);
    DenomH = max(DenomH, tiny);

    HB = NumerH ./ DenomH;
    HB = max(HB, 0);

    empty_cols = ~any(MB, 1);
    if any(empty_cols)
        HB(:, empty_cols) = 0;
    end

    H(:, idx) = HB;
end

end

% =========================================================================
function [NumerW, DenomW] = compute_W_terms_masked_blockwise_local(W, H, X0, M, blockN)
%% Compute masked multiplicative-update terms for W in column blocks
%
% Numerator:
%   X0 * H'
%
% Denominator:
%   (M .* (W * H)) * H'
%
% No D-by-D matrices are formed.

[D, K] = size(W);
N = size(X0, 2);

NumerW = zeros(D, K);
DenomW = zeros(D, K);

for first = 1:blockN:N
    last = min(first + blockN - 1, N);
    idx = first:last;

    XB = X0(:, idx);      % D-by-b
    HB = H(:, idx);       % K-by-b
    MB = M(:, idx);       % D-by-b logical

    NumerW = NumerW + XB * HB';

    YB = W * HB;          % D-by-b
    YB(~MB) = 0;          % apply observation mask
    DenomW = DenomW + YB * HB';
end

end

% =========================================================================
function blockN = choose_block_size_local(D, N)
%% Simple heuristic for column block size
%
% The goal is to limit the size of D-by-blockN temporary matrices.
% You can tune target_numel below if needed.

target_numel = 2e6;   % heuristic temporary size target
blockN = floor(target_numel / max(D, 1));
blockN = max(blockN, 1);
blockN = min(blockN, N);

end

% =========================================================================
function [W, H] = reorder_components_local(W, H)
%% Reorder components by descending component energy

K = size(H, 1);

hlen = sqrt(sum(H.^2, 2));
zero_comp = (hlen == 0);

if any(zero_comp)
    warning('opnmf_nan:LowRank', ...
        'Only %d out of %d components have nonzero energy in H.', ...
        K - sum(zero_comp), K);
    hlen(zero_comp) = 1;
end

Wh = bsxfun(@times, W, hlen');
[~, idx] = sort(sum(Wh.^2, 1), 'descend');

W = W(:, idx);
H = H(idx, :);

end

% =========================================================================
function save_intermediate_local(outputdir, iter, D, K, W)
%% Save reduced-space W for restart / recovery

try
    save(fullfile(outputdir, 'IntermResultsExtractBases.mat'), ...
        'iter', 'D', 'K', 'W', '-v7.3');
catch ME
    warning('opnmf_nan:saveFail', ...
        'Failed to save intermediate results: %s', ME.message);
end

end

% =========================================================================
function [W, H] = NNDSVD_nanaware_local(A, k, flag)
%% NaN-aware NNDSVD wrapper
%
% Missing entries are zero-filled. Rows are additionally rescaled by the
% inverse square root of their observed counts before SVD so that rows with
% many missing entries are not systematically underrepresented during the
% SVD-based initialization step.

tiny = 1e-16;

if any(A(~isnan(A)) < 0)
    error('NNDSVD_nanaware_local:negativeInput', ...
        'Observed entries must be nonnegative.');
end

M = ~isnan(A);

A0 = A;
A0(~M) = 0;

row_count = sum(M, 2);
row_count_safe = max(row_count, 1);

scale = 1 ./ sqrt(row_count_safe);
A_scaled = bsxfun(@times, A0, scale);

[W0, H0] = NNDSVD_local(A_scaled, k, flag);

W = bsxfun(@rdivide, W0, scale);
H = H0;

W(~isfinite(W)) = 0;
H(~isfinite(H)) = 0;

W = max(W, tiny);
H = max(H, tiny);

if norm(W, 'fro') == 0 || norm(H, 'fro') == 0
    avgA = mean(A0(M));
    if isempty(avgA) || ~isfinite(avgA) || avgA <= 0
        avgA = 1;
    end
    W = max(avgA * rand(size(A, 1), k), tiny);
    H = max(W' * A0, tiny);
end

end

% =========================================================================
function [W, H] = NNDSVD_local(A, k, flag)
%% NNDSVD / NNDSVDa / NNDSVDar initializer
%
% flag:
%   0 -> NNDSVD
%   1 -> NNDSVDa
%   2 -> NNDSVDar
%   3 -> NNDSVD using randomized SVD backend when available

tiny = 1e-16;

if any(A(:) < 0)
    error('NNDSVD_local:negativeInput', ...
        'The input matrix contains negative elements.');
end

[m, n] = size(A);
W = zeros(m, k);
H = zeros(k, n);

if nnz(A) == 0
    W = max(rand(m, k), tiny);
    H = max(rand(k, n), tiny);
    return;
end

% Compute a rank-k SVD using a robust backend cascade:
%   randomized SVD (if requested and available) -> svds -> full svd
try
    use_randpca = (flag == 3) && ...
        (exist('randpca', 'file') == 2 || exist('randpca', 'file') == 6);

    if use_randpca
        l = max(3 * k, 20);
        [U, S, V] = randpca(A, k, true, 8, l);

    elseif k < min(m, n)
        [U, S, V] = svds(A, k);

    else
        [Ufull, Sfull, Vfull] = svd(full(A), 'econ');
        U = Ufull(:, 1:k);
        S = Sfull(1:k, 1:k);
        V = Vfull(:, 1:k);
    end
catch
    [Ufull, Sfull, Vfull] = svd(full(A), 'econ');
    U = Ufull(:, 1:k);
    S = Sfull(1:k, 1:k);
    V = Vfull(:, 1:k);
end

U = real(U);
S = real(S);
V = real(V);

sigma1 = max(S(1,1), 0);
W(:,1) = sqrt(sigma1) * abs(U(:,1));
H(1,:) = sqrt(sigma1) * abs(V(:,1)');

for i = 2:k
    uu = U(:, i);
    vv = V(:, i);

    uup = pos_local(uu);
    uun = neg_local(uu);
    vvp = pos_local(vv);
    vvn = neg_local(vv);

    n_uup = norm(uup);
    n_uun = norm(uun);
    n_vvp = norm(vvp);
    n_vvn = norm(vvn);

    termp = n_uup * n_vvp;
    termn = n_uun * n_vvn;
    sigma = max(S(i, i), 0);

    if termp >= termn
        if n_uup > 0 && n_vvp > 0
            W(:, i) = sqrt(sigma * termp) * (uup / n_uup);
            H(i, :) = sqrt(sigma * termp) * (vvp' / n_vvp);
        end
    else
        if n_uun > 0 && n_vvn > 0
            W(:, i) = sqrt(sigma * termn) * (uun / n_uun);
            H(i, :) = sqrt(sigma * termn) * (vvn' / n_vvn);
        end
    end
end

W(W < 0) = 0;
H(H < 0) = 0;

avgA = mean(A(:));
if ~isfinite(avgA)
    avgA = 0;
end

switch flag
    case 1   % NNDSVDa
        W(W == 0) = avgA;
        H(H == 0) = avgA;

    case 2   % NNDSVDar
        idxW = (W == 0);
        idxH = (H == 0);

        if nnz(idxW) > 0
            W(idxW) = avgA * rand(nnz(idxW), 1) / 100;
        end
        if nnz(idxH) > 0
            H(idxH) = avgA * rand(nnz(idxH), 1) / 100;
        end
end

% Final floor to reduce exact-zero locking
W = max(W, tiny);
H = max(H, tiny);

end

% =========================================================================
function Ap = pos_local(A)
Ap = (A >= 0) .* A;
end

% =========================================================================
function Am = neg_local(A)
Am = (A < 0) .* (-A);
end
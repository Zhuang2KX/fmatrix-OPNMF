function [X_nonneg, X_ips, shiftConst, subjMean] = get_ipsatize(data, revIdx, reverseFirst, scaleRange)
% IPSATIZE_AND_SHIFT_NONNEG Ipsatize questionnaire responses and shift to strictly positive.
%
% Inputs
%   data         : [nSubj x nItems] numeric matrix. Missing values must be NaN.
%   revIdx       : Items to reverse-key (negatively keyed items). Logical mask or numeric indices.
%   reverseFirst : (optional) logical scalar selecting preprocessing order.
%                  false (default): ipsatize first, then apply sign-flip on revIdx columns
%                  true           : reverse-key raw data first using (min+max)-x, then ipsatize
%   scaleRange   : (optional) [minVal, maxVal] of the response scale (e.g., [1 5]).
%                  If omitted, inferred from data (may be risky if observed range is truncated).
%
% Outputs
%   X_nonneg     : shifted matrix with all non-missing entries strictly > 0 (NaNs preserved).
%   X_ips        : preprocessed (ipsatized ± reverse-keyed) matrix (can be negative; NaNs preserved).
%   shiftConst   : scalar added to X_ips to obtain X_nonneg.
%   subjMean     : [nSubj x 1] within-subject mean used for ipsatization (omitnan).

    if nargin < 3 || isempty(reverseFirst)
        reverseFirst = false;
    end

    % Ensure numeric double
    X0 = data;
    if ~isfloat(X0), X0 = double(X0); end

    % Validate / normalize revIdx
    nItems = size(X0, 2);
    if isempty(revIdx)
        revMask = false(1, nItems);
    elseif islogical(revIdx)
        if numel(revIdx) ~= nItems
            error('revIdx logical vector must have length equal to nItems.');
        end
        revMask = revIdx(:).';
    elseif isnumeric(revIdx)
        revMask = false(1, nItems);
        revMask(revIdx) = true;
    else
        error('revIdx must be [] or a logical vector or a numeric index vector.');
    end
    cols = find(revMask);

    % ---- Scale detection (only needed for reverseFirst=true branch) ----
    if nargin < 4 || isempty(scaleRange)
        detectedMin = min(X0(:), [], 'omitnan');
        detectedMax = max(X0(:), [], 'omitnan');
        if isempty(detectedMin) || isnan(detectedMin) || isnan(detectedMax)
            error('Data is all NaN; cannot detect scale.');
        end
        scaleSum = detectedMin + detectedMax;
    else
        if numel(scaleRange) ~= 2
            error('scaleRange must be a 2-element vector: [minVal maxVal].');
        end
        scaleSum = sum(scaleRange);
    end

    % ---- Main preprocessing ----
    if reverseFirst
        % Reverse-key raw data FIRST, then ipsatize.
        X1 = X0;
        if ~isempty(cols)
            T = X1(:, cols);
            m = ~isnan(T);
            T(m) = scaleSum - T(m);
            X1(:, cols) = T;
        end

        subjMean = mean(X1, 2, 'omitnan');
        X_ips = X1 - subjMean;
        X_ips(isnan(X1)) = NaN;

    else
        % Ipsatize FIRST, then flip direction of revIdx items in centered space.
        subjMean = mean(X0, 2, 'omitnan');
        X_ips = X0 - subjMean;
        X_ips(isnan(X0)) = NaN;

        if ~isempty(cols)
            T = X_ips(:, cols);
            m = ~isnan(T);
            T(m) = -T(m);
            X_ips(:, cols) = T;
        end
    end

    % ---- Nonnegative shift for NMF/OPNMF ----
    globalMin = min(X_ips(:), [], 'omitnan');
    if isempty(globalMin) || isnan(globalMin)
        error('All entries are NaN after preprocessing.');
    end

    finiteVals = X_ips(~isnan(X_ips));
    scaleVal = max(abs(finiteVals));
    if isempty(scaleVal) || scaleVal == 0, scaleVal = 1; end
    epsAuto = max(1e-8, 1e-12 * scaleVal);

    shiftConst = -globalMin + epsAuto;

    X_nonneg = X_ips + shiftConst;
    X_nonneg(isnan(X_ips)) = NaN;

    if any(X_nonneg(~isnan(X_nonneg)) <= 0)
        error('Shift failed: non-positive values remain.');
    end
end
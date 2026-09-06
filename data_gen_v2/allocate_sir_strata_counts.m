function counts = allocate_sir_strata_counts(total_count, weights)
%% allocate_sir_strata_counts — 按最大余数法得到确定性的整数分层配额

if ~isnumeric(total_count) || ~isscalar(total_count) || ~isfinite(total_count) || ...
        total_count < 0 || total_count ~= fix(total_count)
    error('allocate_sir_strata_counts:TotalCount', ...
        'total_count 必须是非负整数。');
end
if ~isnumeric(weights) || ~isreal(weights) || isempty(weights) || ...
        any(~isfinite(weights)) || any(weights <= 0) || abs(sum(weights) - 1) > 1e-12
    error('allocate_sir_strata_counts:Weights', ...
        'weights 必须是有限正数且总和为 1。');
end
weights = weights(:).';
raw_counts = total_count * weights;
counts = floor(raw_counts);
fractions = raw_counts - counts;
remaining = total_count - sum(counts);
for idx = 1:remaining
    largest_fraction = max(fractions);
    selected = find(abs(fractions - largest_fraction) <= 1e-12, 1, 'first');
    counts(selected) = counts(selected) + 1;
    fractions(selected) = -Inf;
end
end

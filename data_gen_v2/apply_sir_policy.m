function decision = apply_sir_policy(sir_audit, sir_mode, bin_edges_db, ...
    accepted_counts, target_counts)
%% apply_sir_policy — 只决定候选接受/重抽，不接触或修改音频

if ~(ischar(sir_mode) || (isstring(sir_mode) && isscalar(sir_mode)))
    error('apply_sir_policy:Mode', 'sir_mode 必须是字符串。');
end
mode = lower(strtrim(char(sir_mode)));
decision = struct( ...
    'accept_candidate', true, ...
    'reject_candidate', false, ...
    'resample_candidate', false, ...
    'gain_adjustment_applied', false, ...
    'stratum', 0, ...
    'reason', 'measure_only');

if strcmp(mode, 'measure_only')
    return;
end
if ~strcmp(mode, 'stratified')
    error('apply_sir_policy:Mode', ...
        '未知 sir_mode=''%s''；仅支持 measure_only 或 stratified。', mode);
end
validate_policy_inputs(bin_edges_db, accepted_counts, target_counts);

decision.accept_candidate = false;
decision.reject_candidate = true;
decision.resample_candidate = true;
decision.reason = 'invalid_overlap_band_sir';
if ~isstruct(sir_audit) || ...
        ~isfield(sir_audit, 'sir_overlap_band_valid') || ...
        ~isfield(sir_audit, 'sir_rx_overlap_band_db') || ...
        ~isscalar(sir_audit.sir_overlap_band_valid) || ...
        sir_audit.sir_overlap_band_valid ~= 1 || ...
        ~isscalar(sir_audit.sir_rx_overlap_band_db) || ...
        ~isfinite(sir_audit.sir_rx_overlap_band_db)
    return;
end

sir_db = sir_audit.sir_rx_overlap_band_db;
if sir_db < bin_edges_db(1) || sir_db > bin_edges_db(end)
    decision.reason = 'outside_range';
    return;
end

last_bin = numel(bin_edges_db) - 1;
stratum = find(sir_db >= bin_edges_db(1:end-1) & ...
    sir_db < bin_edges_db(2:end), 1, 'first');
if isempty(stratum) && sir_db == bin_edges_db(end)
    stratum = last_bin;
end
if isempty(stratum)
    decision.reason = 'outside_range';
    return;
end
decision.stratum = stratum;
if accepted_counts(stratum) >= target_counts(stratum)
    decision.reason = 'stratum_full';
    return;
end

decision.accept_candidate = true;
decision.reject_candidate = false;
decision.resample_candidate = false;
decision.reason = 'accepted';
end


function validate_policy_inputs(bin_edges_db, accepted_counts, target_counts)
if ~isnumeric(bin_edges_db) || ~isreal(bin_edges_db) || ...
        numel(bin_edges_db) < 2 || any(~isfinite(bin_edges_db)) || ...
        any(diff(bin_edges_db) <= 0)
    error('apply_sir_policy:Edges', ...
        'bin_edges_db 必须是严格递增的有限数值向量。');
end
number_of_bins = numel(bin_edges_db) - 1;
if ~isnumeric(accepted_counts) || ~isnumeric(target_counts) || ...
        numel(accepted_counts) ~= number_of_bins || ...
        numel(target_counts) ~= number_of_bins || ...
        any(~isfinite(accepted_counts)) || any(~isfinite(target_counts)) || ...
        any(accepted_counts < 0) || any(target_counts < 0) || ...
        any(accepted_counts ~= fix(accepted_counts)) || ...
        any(target_counts ~= fix(target_counts)) || ...
        any(accepted_counts > target_counts)
    error('apply_sir_policy:Counts', ...
        'accepted_counts/target_counts 必须是与分层匹配的有效整数计数。');
end
end

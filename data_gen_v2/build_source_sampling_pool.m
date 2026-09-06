function pool = build_source_sampling_pool(wavs, recording_map)
%% build_source_sampling_pool — 将 split 内模板按原始录音分组
%
% 只使用传入 split 的 wavs；recording_map 可以包含其他 split 的映射，但
% 不会把其模板加入 pool。录音键排序，保证相同输入下采样顺序可复现。

if ~isstruct(wavs) || ~isfield(wavs, 'name')
    error('build_source_sampling_pool:InvalidWavs', 'wavs 必须是含 name 字段的 dir 结构体。');
end
if ~isa(recording_map, 'containers.Map')
    error('build_source_sampling_pool:InvalidMap', 'recording_map 必须是 containers.Map。');
end
if isempty(wavs)
    error('build_source_sampling_pool:EmptyPool', '模板池不能为空。');
end

file_recordings = cell(1, numel(wavs));
grouped = containers.Map('KeyType', 'char', 'ValueType', 'any');
for idx = 1:numel(wavs)
    name = wavs(idx).name;
    if ~recording_map.isKey(name)
        error('build_source_sampling_pool:MissingMapping', ...
            '录音映射缺少 split 模板 %s。', name);
    end
    recording = recording_map(name);
    file_recordings{idx} = recording;
    if grouped.isKey(recording)
        grouped(recording) = [grouped(recording), idx];
    else
        grouped(recording) = idx;
    end
end

recording_keys = sort(keys(grouped));
file_indices_by_recording = cell(1, numel(recording_keys));
template_counts = zeros(1, numel(recording_keys));
for idx = 1:numel(recording_keys)
    indices = grouped(recording_keys{idx});
    file_indices_by_recording{idx} = indices;
    template_counts(idx) = numel(indices);
end

pool = struct();
pool.wavs = wavs;
pool.file_recordings = file_recordings;
pool.recording_keys = recording_keys;
pool.file_indices_by_recording = file_indices_by_recording;
pool.template_counts = template_counts;
pool.n_templates = numel(wavs);
pool.n_recordings = numel(recording_keys);
end

function [file_t, recording_key, file_index, recording_index] = ...
    sample_source_from_pool(pool, mode)
%% sample_source_from_pool — 模板均匀或两阶段录音均匀采样
%
% recording_uniform 使用一个 U(0,1) 随机数同时选择录音及录音内模板：
% floor(U*R) 选择录音，条件小数部分选择该录音内模板。这样每个候选源
% 仍只消耗一个主随机数，避免仅因两阶段实现额外推进全局随机流。

if ~(ischar(mode) || (isstring(mode) && isscalar(mode)))
    error('sample_source_from_pool:InvalidMode', 'mode 必须是字符串标量。');
end
mode = lower(strtrim(char(mode)));

switch mode
    case 'template_uniform'
        file_index = randi(pool.n_templates);
        recording_key = pool.file_recordings{file_index};
        recording_index = find(strcmp(pool.recording_keys, recording_key), 1);
    case 'recording_uniform'
        u = rand();
        scaled = u * pool.n_recordings;
        recording_index = min(floor(scaled) + 1, pool.n_recordings);
        within_recording = scaled - floor(scaled);
        candidates = pool.file_indices_by_recording{recording_index};
        local_index = min(floor(within_recording * numel(candidates)) + 1, ...
                          numel(candidates));
        file_index = candidates(local_index);
        recording_key = pool.recording_keys{recording_index};
    otherwise
        error('sample_source_from_pool:InvalidMode', ...
            '未知 source_sampling_mode=%s。', mode);
end

file_t = pool.wavs(file_index);
if ~strcmp(pool.file_recordings{file_index}, recording_key)
    error('sample_source_from_pool:InternalMismatch', '模板与录音分组不一致。');
end
end

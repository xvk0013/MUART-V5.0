function test_d1_recording_uniform()
%% test_d1_recording_uniform — 不读音频的合成采样测试

wavs = repmat(struct('name', '', 'folder', ''), 1, 10);
mapping = containers.Map('KeyType', 'char', 'ValueType', 'char');
for idx = 1:10
    wavs(idx).name = sprintf('%05d.wav', idx);
    wavs(idx).folder = 'synthetic';
    if idx == 1, recording = 'recording_a'; else, recording = 'recording_b'; end
    mapping(wavs(idx).name) = recording;
end
pool = build_source_sampling_pool(wavs, mapping);
assert(pool.n_templates == 10 && pool.n_recordings == 2);
assert(isequal(pool.template_counts, [1 9]));

n = 20000;
rng(6101);
template_b = 0;
for idx = 1:n
    [~, recording] = sample_source_from_pool(pool, 'template_uniform');
    template_b = template_b + strcmp(recording, 'recording_b');
end
template_b_rate = template_b / n;
assert(abs(template_b_rate - 0.9) < 0.02, ...
    'template_uniform 未按模板数量加权录音。');

rng(6101);
recording_b = 0;
file_counts = zeros(1,10);
for idx = 1:n
    [~, recording, file_index] = sample_source_from_pool(pool, 'recording_uniform');
    recording_b = recording_b + strcmp(recording, 'recording_b');
    file_counts(file_index) = file_counts(file_index) + 1;
end
recording_b_rate = recording_b / n;
assert(abs(recording_b_rate - 0.5) < 0.02, ...
    'recording_uniform 未实现录音等概率。');
conditional_b = file_counts(2:10) / sum(file_counts(2:10));
assert(max(abs(conditional_b - 1/9)) < 0.02, ...
    '录音内模板不是近似均匀。');

% recording_uniform 每次只能消耗一个主随机数。
rng(1234);
sample_source_from_pool(pool, 'recording_uniform');
after_sample = rand();
rng(1234);
rand();
expected_after = rand();
assert(after_sample == expected_after, ...
    '两阶段采样额外推进了全局随机流。');

bad_mapping = containers.Map('KeyType', 'char', 'ValueType', 'char');
bad_mapping(wavs(1).name) = 'recording_a';
try
    build_source_sampling_pool(wavs, bad_mapping);
    error('test_d1_recording_uniform:ExpectedFailure', '缺失映射未触发失败。');
catch exc
    assert(strcmp(exc.identifier, 'build_source_sampling_pool:MissingMapping'));
end

fprintf(['PASS: template_uniform recording_b=%.4f；' ...
         'recording_uniform recording_b=%.4f；录音内模板均匀。\n'], ...
    template_b_rate, recording_b_rate);
end

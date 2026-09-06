function cfg = bp12_mainline_config()
%% bp12_mainline_config — MUART V5.0 冻结的 bp12 主线配置
%
% 本函数是 Phase 1/Phase 3 主入口与配置校验器共享的唯一主线常量源。
% bp12 主线为 BPF-only，G=1，不是启用 Wiener 的结果。
%
% ER 主线策略: natural。逐录音 SL 原样进入声呐方程，不做人为 ER 截断。

data_gen_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(data_gen_dir);
data_root = resolve_muart_data_root();

cfg.data_gen_dir = data_gen_dir;
cfg.project_root = project_root;
cfg.data_root = data_root;

% Phase 1: bp12 BPF-only 模板。
cfg.bpf_hz = [12 7900];
cfg.g_floor_db = [0 0];
cfg.template_dir_name = 'data_sl_templates_bp12';
cfg.template_root = fullfile(data_root, cfg.template_dir_name);

% Phase 2/3: 冻结的 bp12 数据基线。
cfg.split_dir_name = 'data_sl_split_bp12';
cfg.split_root = fullfile(data_root, cfg.split_dir_name);
cfg.output_root = fullfile(data_root, 'data_cargo_sl_sir15');
cfg.pilot_sl_table_path = fullfile(project_root, 'data_cargo_sl', 'sl_tables.txt');
cfg.sl_mean = [183 183 180.5];
cfg.sl_std = 3;
cfg.rng_sl_seed = 1000;
cfg.cond_winds = [2.5];
cfg.cond_names = {'SS2_bp12'};
cfg.sl_table_name = 'sl_tables.txt';
cfg.shipping_factor = 0.2;
cfg.er_policy = 'natural';
cfg.sir_mode = 'stratified';
cfg.audit_band = [20 4000];
cfg.sir_bin_edges_db = [0 5 10 15];
cfg.sir_bin_weights = [0.45 0.35 0.20];
cfg.sir_max_attempt_factor = 20;

% 第 5 阶段 pilot：默认不随生产入口运行；直接调用 pilot 入口才显式执行。
cfg.pilot_enabled = false;
cfg.pilot_split = 'Train';
cfg.pilot_samples_per_pair = 200;
cfg.pilot_seed = 42003;
cfg.pilot_report_root = fullfile(data_gen_dir, 'pilot_reports');

% 标签位 [Cargo, Tanker, Tug] 与七种样本类型。
cfg.class_ids = [0 1 3];
cfg.class_names = {'Cargo', 'Tanker', 'Tug'};
cfg.combos = { ...
    [],    'noise'; ...
    [0],   '0'; ...
    [1],   '1'; ...
    [2],   '2'; ...
    [0 1], '0_1'; ...
    [0 2], '0_2'; ...
    [1 2], '1_2'};
cfg.totals = [11490 2463 2463];             % Train/Test/Val
cfg.split_types = {'Train', 'Test', 'Val'};
cfg.props = struct('noise', 1512/11490, ...
                   'single', 3174/11490, ...
                   'double', 6804/11490);
end

%% gen_arr — BELLHOP 宽带多频点自动批处理运行脚本
%
% 功能: 以 template_pos1.env 为模板, 自动修改频率行,
%       生成 80 个不同频率的 .env 环境文件,
%       逐频点调用 bellhop.exe 运行声场仿真, 生成 .arr 到达文件。
%
% 频率覆盖:
%   50:100:7950 Hz (80 个算术中心频点),
%   覆盖了 17kHz 重采样后的全部有效带宽 (0~8 kHz 通带)。
%   每个频点独立运行一次 BELLHOP, 产生包含该频率下所有本征声线
%   复数幅度 (A) 和群延迟 (delay) 的 .arr 文件。
%
% 并行策略:
%   本脚本为串行版本 (逐频点运行, 总计约 80×30s ≈ 40 分钟)。
%   若需加速, 可将 80 个频点分成 N 组, 每组在独立的 MATLAB
%   进程中并行执行 (BELLHOP 是无状态的命令行程序, 无冲突)。
%
% 运行前提:
%   - bellhop.exe 在 MATLAB 路径中 (或与 gen_arr.m 同目录)
%   - template_pos1.env 存在且包含完整的声速剖面/海底/几何配置
%   - .ssp / .bty / .brc / .ati / .trc / .sbp 辅助文件存在
%
% 输出:
%   Pos1Azi1freq50Hz.arr ... Pos1Azi1freq7950Hz.arr   (80个到达文件)
%   prt_logs/ Pos1Azi1freq50Hz.prt ...               (80个运行日志)
%   .env / .shd / .ssp 等临时文件在运行后被自动清理
%
% 依赖:
%   bellhop.m           BELLHOP 可执行文件包装器
%   bellhop.exe         AcTUP / OALIB 水声工具箱
%   template_pos1.env   环境模板文件
%   .ssp / .bty / .brc  辅助配置文件

%% ===== 配置 =====
f_centers = 50:100:7950;                             % 80 个宽带算术中心频点
base_env = 'template_pos1';                           % 模板文件名前缀 (不含扩展名)

%% ===== 模板文件检查 =====
if ~exist([base_env '.env'], 'file')
    error('未找到 %s.env，请检查 BELLHOP 模板文件！', base_env);
end

total_runs = length(f_centers);
t_start = tic;                                       % 计时开始

%% ===== 创建 prt 日志存放目录 =====
prt_log_dir = 'prt_logs';
if ~exist(prt_log_dir, 'dir')
    mkdir(prt_log_dir);
end

%% ===== 统计变量 =====
success_count = 0;                                   % 成功计数器
fail_list = [];                                      % 失败频点列表

%% ===== 主循环: 逐频点修改 .env → 运行 BELLHOP → 验证输出 =====
for i = 1:total_runs
    freq = f_centers(i);
    fprintf('正在处理频率 [%d/%d]: %.0f Hz...\n', i, total_runs, freq);
    target_env = sprintf('Pos1Azi1freq%dHz', freq);  % 目标 .env 文件名 (频率嵌入)

    ok = run_bellhop_ultimate(base_env, target_env, freq, prt_log_dir);
    if ok
        success_count = success_count + 1;
    else
        fail_list(end+1) = freq;                     % 记录失败频点
    end
end

t_elapsed = toc(t_start);                            % 总耗时

%% ===== 结果报告 =====
if isempty(fail_list)
    fprintf('\n批处理完成！成功生成全部 %d/%d 个 .arr 文件。\n', success_count, total_runs);
    fprintf('总耗时: %.1f 分钟\n', t_elapsed / 60);
else
    fprintf('\n批处理完成！成功 %d/%d，失败频点: %s\n', ...
        success_count, total_runs, mat2str(fail_list));
    fprintf('总耗时: %.1f 分钟\n', t_elapsed / 60);
end

%% ===== 局部函数: 单频点 BELLHOP 运行 =====
function ok = run_bellhop_ultimate(base_name, target_name, new_freq, prt_log_dir)
% run_bellhop_ultimate — 单频点 BELLHOP 完整执行流程
%
% 步骤:
%   ① 读取模板 .env → ② 修改第2行频率值 → ③ 写入新 .env
%   → ④ 复制 .brc/.ssp/.bty 等辅助文件 → ⑤ 调用 bellhop.exe
%   → ⑥ 验证 .arr 生成 → ⑦ 移动 .prt 到日志目录 → ⑧ 清理临时文件

    % ----- 第1步: 读取模板 .env 文件 -----
    fid_in = fopen([base_name '.env'], 'rt');
    if fid_in == -1
        warning('无法打开模板文件 %s.env', base_name);
        ok = false;
        return;
    end

    lines = {};
    tline = fgetl(fid_in);
    while ischar(tline)
        lines{end+1} = tline;                       
        tline = fgetl(fid_in);
    end
    fclose(fid_in);

    % ----- 第2步: 修改第2行频率值, 保留原名及注释 -----
    orig_line2 = lines{2};
    excl_idx = strfind(orig_line2, '!');             % 查找 Fortran 注释分隔符 !

    if ~isempty(excl_idx)
        lines{2} = sprintf('  %.2f     %s', new_freq, orig_line2(excl_idx:end));
    else
        lines{2} = sprintf('  %.2f     ! Frequency (Hz)', new_freq);
    end

    % ----- 第3步: 写入新 .env 文件 -----
    fid_out = fopen([target_name '.env'], 'wt');
    if fid_out == -1
        warning('无法写入 %s.env', target_name);
        ok = false;
        return;
    end
    for i = 1:length(lines), fprintf(fid_out, '%s\n', lines{i}); end
    fclose(fid_out);

    % ----- 第4步: 复制 .brc/.ssp/.bty 等辅助文件 -----
    % BELLHOP 读取同名的 .ssp (声速剖面), .bty (海底地形), .brc (底反射系数)
    input_exts = {'.brc', '.ssp', '.bty', '.ati', '.trc', '.sbp'};
    for j = 1:length(input_exts)
        if exist([base_name input_exts{j}], 'file')
            [s, m] = copyfile([base_name input_exts{j}], [target_name input_exts{j}]);
            if ~s
                warning('复制 %s 失败: %s', input_exts{j}, m);
                ok = false;
                return;
            end
        end
    end

    % ----- 第5步: 调用 BELLHOP 运行声场仿真 -----
    try
        bellhop(target_name);
    catch ME
        warning(ME);
        ok = false;
        return;
    end

    % ----- 第6步: 验证 .arr 文件是否生成 -----
    if ~exist([target_name '.arr'], 'file')
        warning('%s: BELLHOP 运行完成但未生成 .arr 文件', target_name);
        ok = false;
    else
        ok = true;
    end

    % ----- 第7步: 保留 .prt 日志至 prt_logs/ 目录 -----
    prt_file = [target_name '.prt'];
    if exist(prt_file, 'file')
        [s, m] = movefile(prt_file, fullfile(prt_log_dir, [target_name '.prt']));
        if ~s
            warning('移动 .prt 日志失败: %s', m);
        end
    end

    % ----- 第8步: 清理临时生成的环境文件 (仅保留 .arr) -----
    temp_exts = {'.env', '.shd', '.brc', '.ssp', '.bty', '.ati', '.trc', '.sbp'};
    for j = 1:length(temp_exts)
        if exist([target_name temp_exts{j}], 'file')
            delete([target_name temp_exts{j}]);
        end
    end
end
function [Arr, Pos] = read_arrivals_asc(fname)
%% read_arrivals_asc — BELLHOP 2D .arr 到达文件解析
%
% 格式按 _probe_arr2.py 实证解析移植 (与 arr_utils.py read_arrivals 一致,
% Python 端已用 concat_v2 已存参考通道验证 corr=1.0000):
%
% 文件结构:
%   头 5 行: '2D' / freq / [NSz + 源深...] / [NRz + 接收深...] / [NRr + 距离...]
%   之后 NSz 个 section (源深), 每个 = 1 行额外整数 (该 section 最大到达数, 跳过)
%     + NRz×NRr 个块; 块序 = 接收深主序 × 距离次序 (bi = (nz-1)*NRr + k_r)
%   每块: 1 行到达数 n + n 行 arrival (8 列):
%     [amp, phase_deg, tau, x, SrcAz, RcvAz, nTop, nBot]
%   复幅度 A = amp · exp(1j · phase_deg · π/180)
%
% 输入:
%   fname - .arr 文件路径
% 输出:
%   Arr  - NRr×NRz×NSz struct array, 字段 .A (复幅度向量) / .delay (延迟, 秒)
%          索引约定: Arr(k_r, nz, n_d) = 距离 k_r × 接收深 nz × 源深 section n_d
%   Pos  - .s.z (源深数组, m) / .r.r (距离数组, m)
%
% 依赖: 无 (纯 MATLAB 内置函数)

    txt = fileread(fname);
    lines = regexp(txt, '\r?\n', 'split');
    keep = ~cellfun(@(l) isempty(strtrim(l)), lines);
    lines = lines(keep);
    lines = cellfun(@strtrim, lines, 'UniformOutput', false);

    % ===== 头部解析 =====
    tok_s  = strsplit(lines{3});          % [NSz, z_src...]
    tok_r  = strsplit(lines{4});          % [NRz, z_recv...]
    tok_rr = strsplit(lines{5});          % [NRr, r...]
    NSz = str2double(tok_s{1});
    NRz = str2double(tok_r{1});
    NRr = str2double(tok_rr{1});
    Pos.s.z = str2double(tok_s(2:end));      % 源深 (m)
    Pos.r.z = str2double(tok_r(2:end));      % 接收深 (m)
    Pos.r.r = str2double(tok_rr(2:end));     % 距离 (m)

    % ===== 主体: section (源深) → 接收深 × 距离 块 =====
    Arr = repmat(struct('A', [], 'delay', []), NRr, NRz, NSz);
    pos = 6;
    for n_d = 1:NSz
        pos = pos + 1;                    % section 额外整数 (最大到达数), 跳过
        for nz = 1:NRz
            for k_r = 1:NRr
                n = str2double(lines{pos});
                pos = pos + 1;
                if n > 0
                    rows = zeros(n, 8);
                    for k = 1:n
                        rows(k, :) = str2double(strsplit(lines{pos}));
                        pos = pos + 1;
                    end
                    Arr(k_r, nz, n_d).A = rows(:, 1) .* exp(1j * rows(:, 2) * pi / 180);
                    Arr(k_r, nz, n_d).delay = rows(:, 3);
                end
            end
        end
    end
    % pos 是 1-based "下一未读行" 指针: 全部行恰好耗尽时 pos = numel(lines) + 1
    % (与 arr_utils.py 的 0-based 断言 pos == len(lines) 严格等价;
    %  80 个 .arr 全部实证校验通过: 5 头行 + 3×(1+12) 结构行 + 全部到达行)
    assert(pos == numel(lines) + 1, ...
        '%s: 解析未耗尽 (已读 %d/%d 行) — .arr 格式与预期不符', fname, pos - 1, numel(lines));
end

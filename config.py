r"""
M-UART v4.0 配置 — TF-ConvNeXt + 双通道特征 (LOFAR 0-1kHz + 宽带 1-4kHz) + DEMON

主干线 (dual, concat_v2 数据, EMR@0.5):
  单模型 3-seed 0.7733 → Gen1 蒸馏 0.7812 → Gen2 蒸馏 0.7854 (纪录)
  集成 dual+Gen1 6模型 0.7933 → +TC 数量约束解码 0.8193 (SOTA, eval_final.py)

已关闭并删除的分支 (结论见 REPORT_final.md):
  CWT/宽带全替换/tri 三通道/dual-hires、Query Anchoring (E1 家族)、
  scpt (SupCon+PT头)、s1/s2/s3 源一致性、comb 梳齿分支、TTA、
  10s 窗口、多域联合、TC 条件化阈值搜索
"""

import os
import torch


class Config:
    """TF-ConvNeXt + dual 双通道 + DEMON + Query Attention 基础配置"""

    # ========== 数据路径 ==========
    # Portable fallback only. Current D1 always receives --data-dir explicitly.
    data_root = os.environ.get(
        'MUART_MODEL_DATA_ROOT',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
    )
    # concat_v2: 拼接数据集 (相遇场景, 随机时间结构), 主线无噪数据集
    # 变体: ..._concat_v2_wenz (接收端 Wenz 噪声) / ..._concat_v2_dense2 (221 信道)
    # 结构: {type}/{split}/{mix,s1,s2,s3}/combined_XXXXX.wav + all_info.txt
    data_combined = os.path.join(data_root, 'deepship_16k_5s_multitarget_ptt_concat_v2')

    # ========== 类别 (3 类) ==========
    # 标签顺序: [Tanker, Passengership, Tug] (config 坐标系)
    class_names = ['Tanker', 'Passengership', 'Tug']
    num_classes = 3
    deepship_id_to_label = {1: 0, 2: 1, 3: 2}

    # main.m 标签位置 → config 索引 (main.m 0=Pass → config 1, 1=Tk → 0, 2=Tug → 2)
    multitarget_pos_to_label = {0: 1, 1: 0, 2: 2}
    # 7 种标签组合 (noise + 3 单 + 3 双; 无三目标)
    multitarget_types = ['noise', '0', '1', '2', '0_1', '0_2', '1_2']
    multitarget_threshold = 0.5

    # ========== 音频 ==========
    sample_rate = 16000
    duration_sec = 5.0
    audio_len = int(sample_rate * duration_sec)  # 80000

    # ========== 主输入特征 (dual 双通道) ==========
    # ch0: LOFAR 0-1kHz, n_fft=8192 (1.95 Hz/bin, 513 bins) — 线谱命根子
    # ch1: 宽带 1-4kHz, n_fft=2048 (7.8 Hz/bin, 385→513 插值) — 高频增量证据
    feature_type = 'dual'
    # [E4 双流融合臂] False 保持历史早期通道堆叠；True 时低频/宽带分别
    # 通过共享权重的单通道 backbone，最终在 Query token 级融合。
    # 该臂必须关闭 SpectrumConditionedGate，确保以 E3 为单变量对照。
    dual_stream_fusion = False
    # global=历史 E1：两个频带共用一个标量 mean/std；
    # per_channel=E2：ch0/ch1 分别使用 Train-only mean/std。
    normalization_mode = 'global'
    lofar_n_fft = 8192
    lofar_hop = 1024
    lofar_win = 8192
    lofar_freq_max_hz = 1000
    lofar_n_bins = int(lofar_freq_max_hz / (sample_rate / lofar_n_fft)) + 1  # 513
    wideband_n_fft = 2048       # ch1 高频通道窗长 (7.8Hz/bin)
    wideband_hop = 1024

    @property
    def lofar_frames(self):
        return (self.audio_len - self.lofar_n_fft) // self.lofar_hop + 1

    @property
    def feature_n_bins(self):
        return self.lofar_n_bins  # lofar / dual 均为 513

    @property
    def feature_frames(self):
        """时间轴尺寸 (STFT center=True: T/hop+1=79, 两通道统一)"""
        return self.audio_len // self.lofar_hop + 1

    # ========== Query Attention 分类头 ==========
    query_num_heads = 4
    query_dropout = 0.2
    query_attn_dropout = 0.1
    pe_max_h = 64                # 频率轴位置编码最大尺寸 (骨干输出 17 bins)
    pe_max_w = 32                # 时间轴位置编码最大尺寸 (骨干输出 5 bins)

    # ========== DEMON 特征 (双分支) ==========
    use_demon = True
    demon_bp_low = 50
    demon_bp_high = 1000
    demon_lp_cutoff = 100
    demon_max_hz = 50
    demon_n_bins = 128
    demon_remove_dc = True
    demon_encoder_dims = [32, 64, 128]
    demon_feat_dim = 128

    # ========== 波形级数据增强 ==========
    use_waveform_aug = True
    aug_noise_snr_db = (5, 20)
    aug_freq_shift_hz = (-10, 10)
    aug_time_stretch = (0.95, 1.05)
    aug_gain_db = (-3, 3)
    aug_freq_shift_hz_single = (-5, 5)    # 单目标温和桶 (保护线谱)
    aug_noise_snr_db_single = (15, 25)
    # white=历史高斯白噪声增强；off=E6 审计臂，在相同抽样槽位和随机数
    # 消耗下执行恒等映射，避免改变其他增强被选中的概率。
    waveform_noise_mode = 'white'

    # 频移审计开关：legacy=保留历史 FFT-roll 实现，仅用于 E0 对照；
    # off=在完全相同的增强抽样位置执行恒等映射，用于 E1 单变量消融。
    # 默认保持 legacy，避免无意改变历史训练行为；新主线是否关闭由实验决定。
    freq_shift_mode = 'legacy'

    # [spd 臂] 速度扰动 (Doppler 对口增强): 训练期 100% 概率施加,
    # 频率等比缩放 ±10% — 谐波结构保持 (物理正确), 对齐 macls 主增强策略;
    # 现有 freq_shift(常量 Hz 平移, 破坏谐波比) 的物理正确替代候选
    use_speed_perturb = False
    aug_speed_perturb = (0.9, 1.1)

    # ========== TF-ConvNeXt 骨干 ==========
    backbone_dims = [64, 128, 256, 512]
    backbone_blocks = [2, 2, 4, 2]
    backbone_strides_freq = [2, 2, 2, 2]   # 频率: 513 → 17
    backbone_strides_time = [2, 2, 2, 1]   # 时间: 79 → 5
    stem_out = 64
    stem_kernel = 7
    tf_time_kernel = 31    # 时间大核 (水平线谱)
    tf_freq_kernel = 11    # 频率核 (谐波间隔)
    expand_ratio = 4
    drop_path_max = 0.1

    # ========== 训练超参数 ==========
    batch_size = 32
    num_epochs = 30
    lr = 3e-4
    weight_decay = 1e-4
    clip_grad_norm = 5.0
    use_amp = True
    # True 时训练只使用 Train/Val，完全不构造或评估 Test DataLoader。
    # 仅通过命令行 --val-only 显式启用，用于模型选择阶段避免反复查看 Test。
    val_only = False
    num_workers = 4
    early_stop_patience = 15
    warmup_epochs = 3
    lr_min = 1e-6

    # ========== SpecAugment ==========
    use_specaug = True
    specaug_n_freq_masks = 2
    specaug_n_time_masks = 2
    specaug_freq_param = 30
    specaug_time_param = 20

    # ========== 损失函数: Multi-label Focal Loss (非对称 + 类别权重) ==========
    focal_alpha = 0.5
    focal_gamma_pos = 2.0        # 正类聚焦 (目标存在)
    focal_gamma_neg = 1.5        # 负类聚焦降低 → 减少虚警
    # 顺序: [Tanker, Passengership, Tug]
    class_weights = [1.0, 1.3, 1.2]

    # ========== Query 正交惩罚 ==========
    ortho_loss_weight = 0.1

    # ========== EMA 权重平均 ==========
    ema_decay = 0.999

    # ========== 频谱条件门控 ==========
    use_spec_gate = True

    # ========== TC 残差多任务分类器 ==========
    tc_loss_weight = 1.0
    res_loss_weight = 0.5
    res_warmup_epochs = 10
    use_stats_pool = False     # [stats-pool 臂] TC/残差头池化: False=mean, True=[attentive-mean‖std]

    # ========== 输出 ==========
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'checkpoints')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ConfigSmall(Config):
    """小模型 ~4.87M 参数 — 主干线默认配置"""
    backbone_dims = [48, 96, 192, 384]
    backbone_blocks = [2, 2, 3, 2]
    backbone_strides_freq = [2, 2, 2, 2]
    pe_max_h = 64
    batch_size = 32
    num_epochs = 30


class ConfigSmallCargo(ConfigSmall):
    """{Tk, Tug, Cargo} 三类变体 — 主线数据集 (data_cargo_v3/dataset)

    四类探针显示 Tk+Tug+Cargo 是 DeepShip 四类最优组合 (防泄漏 LR 0.7315
    vs Pass 组合 0.6740)。数据集由 data_gen/main_cargo_v3.m 生成:
    DeepShip 真实源 + BELLHOP 信道 (源深随机/接收深恒定) + 统一 ER ±5dB
    + Wenz SS2 环境噪声。

    标签位置: type '0'=Cargo, '1'=Tk, '2'=Tug (pos_to_label 不变: {0:1,1:0,2:2})
    类名: [Tanker, Cargo, Tug]; 类权重均匀 [1.0, 1.0, 1.0] (权重重调实验确认)
    """
    data_combined = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'data_cargo_v3', 'dataset')
    class_names = ['Tanker', 'Cargo', 'Tug']
    class_weights = [1.0, 1.0, 1.0]
    # 40/10 控制臂仅 0.7820，压缩余弦周期后性能下降；恢复历史协议。
    num_epochs = 60
    early_stop_patience = 20
    # V5 工作基线：单 seed 审计 E1(0.7824) 优于 legacy E0(0.7714)。
    # legacy 仍可由 --freq-shift-mode legacy 显式复现。
    freq_shift_mode = 'off'


if __name__ == '__main__':
    cfg = ConfigSmall()
    print(f'特征: {cfg.feature_type}, n_bins={cfg.feature_n_bins}, frames={cfg.feature_frames}')
    print(f'类别: {cfg.class_names} ({cfg.num_classes} 类)')
    print(f'骨干: dims={cfg.backbone_dims}, blocks={cfg.backbone_blocks}')
    print(f'TF Block: time_k={cfg.tf_time_kernel}, freq_k={cfg.tf_freq_kernel}')
    print(f'损失: Focal(γ+={cfg.focal_gamma_pos}, γ-={cfg.focal_gamma_neg}) '
          f'+ Ortho(w={cfg.ortho_loss_weight}) + TC(w={cfg.tc_loss_weight})')

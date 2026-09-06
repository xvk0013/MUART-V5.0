import unittest

# Keep the same Windows scientific-runtime import order as train.py/dataset.py.
# Some conda environments otherwise load PyTorch's Intel OpenMP runtime before
# NumPy's runtime and abort with OMP Error #15 during test discovery.
import numpy as np  # noqa: F401 - intentional import-order preflight
import torch
import torch.nn.functional as F

from config_v6a import ConfigV6A, validate_v6a_config
from dataset import multitarget_type_to_label
from model_v6a import (
    LEGAL_SET_MULTI_HOT,
    MUARTV6A,
    conditional_joint_log_probs,
    multihot_to_set_targets,
    set_targets_to_multihot,
)
from train_v6a import conditional_set_loss


class TinyConfig:
    class_names = ["Tanker", "Cargo", "Tug"]
    num_classes = 3
    feature_type = "dual"
    dual_stream_fusion = False
    use_demon = True
    use_spec_gate = False
    feature_n_bins = 65
    feature_frames = 33
    backbone_dims = [8, 12, 16, 24]
    backbone_blocks = [1, 1, 1, 1]
    backbone_strides_freq = [2, 2, 2, 2]
    backbone_strides_time = [2, 2, 2, 1]
    stem_out = 8
    stem_kernel = 3
    tf_time_kernel = 3
    tf_freq_kernel = 3
    expand_ratio = 2
    drop_path_max = 0.0
    demon_encoder_dims = [4, 8]
    demon_feat_dim = 8
    query_num_heads = 4
    query_dropout = 0.0
    query_attn_dropout = 0.0
    pe_max_h = 16
    pe_max_w = 16
    v6a_tc_hidden = 12
    v6a_pair_hidden = 16


class TestV6AConditionalSet(unittest.TestCase):
    def test_config_is_frozen_e8_arm(self):
        config = ConfigV6A()
        validate_v6a_config(config)
        self.assertTrue(config.val_only)
        self.assertEqual(config.normalization_mode, "per_channel")
        self.assertFalse(config.use_spec_gate)
        self.assertEqual(config.waveform_noise_mode, "off")
        self.assertFalse(config.v6a_use_distillation)

    def test_all_legal_targets_round_trip(self):
        labels = LEGAL_SET_MULTI_HOT.clone()
        targets = multihot_to_set_targets(labels)
        self.assertTrue(torch.equal(targets, torch.arange(7)))
        recovered = set_targets_to_multihot(targets)
        self.assertTrue(torch.equal(recovered, labels))

    def test_generator_type_to_legal_set_order(self):
        config = ConfigV6A()
        expected = {
            "noise": 0,
            "1": 1,      # generator position 1 -> Tanker
            "0": 2,      # generator position 0 -> Cargo
            "2": 3,      # generator position 2 -> Tug
            "0_1": 4,    # Tanker + Cargo
            "1_2": 5,    # Tanker + Tug
            "0_2": 6,    # Cargo + Tug
        }
        for type_name, set_index in expected.items():
            label = torch.tensor(
                [multitarget_type_to_label(type_name, config)],
                dtype=torch.float32,
            )
            self.assertEqual(
                int(multihot_to_set_targets(label).item()), set_index
            )

    def test_illegal_triple_is_rejected(self):
        with self.assertRaises(ValueError):
            multihot_to_set_targets(torch.ones(1, 3))

    def test_joint_distribution_is_normalized(self):
        torch.manual_seed(7)
        count = torch.randn(5, 3)
        single = torch.randn(5, 3)
        pair = torch.randn(5, 3)
        log_probs = conditional_joint_log_probs(count, single, pair)
        self.assertEqual(tuple(log_probs.shape), (5, 7))
        self.assertTrue(
            torch.allclose(
                log_probs.exp().sum(dim=1), torch.ones(5), atol=1e-6
            )
        )
        count_probs = F.softmax(count, dim=1)
        self.assertTrue(
            torch.allclose(log_probs[:, 0].exp(), count_probs[:, 0], atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(
                log_probs[:, 1:4].exp().sum(dim=1),
                count_probs[:, 1],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                log_probs[:, 4:7].exp().sum(dim=1),
                count_probs[:, 2],
                atol=1e-6,
            )
        )

    def test_model_forward_loss_and_shared_backbone_gradient(self):
        torch.manual_seed(11)
        model = MUARTV6A(TinyConfig())
        model.train()
        lofar = torch.randn(7, 2, 65, 33)
        demon = torch.randn(7, 1, 32)
        labels = LEGAL_SET_MULTI_HOT.clone()
        outputs = model(lofar, demon)
        self.assertEqual(tuple(outputs["tc_logits"].shape), (7, 3))
        self.assertEqual(tuple(outputs["single_logits"].shape), (7, 3))
        self.assertEqual(tuple(outputs["pair_logits"].shape), (7, 3))
        self.assertEqual(tuple(outputs["set_log_probs"].shape), (7, 7))
        loss, _, targets = conditional_set_loss(outputs, labels)
        self.assertTrue(torch.equal(targets, torch.arange(7)))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grad_sum = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.backbone.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(grad_sum, 0.0)


if __name__ == "__main__":
    unittest.main()

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from config_v6a2 import ConfigV6A2
from model_v6a import MUARTV6A, LEGAL_SET_MULTI_HOT, PAIR_CLASS_INDICES
from model_v6a2 import MUARTV6A2
from dataset import collate_multi
from test_v6a_conditional_set import TinyConfig
import train_v6a2 as training
from experiment_snapshot import collect_resolved_config


class TinyA2(TinyConfig):
    pair_interaction_hidden = 32
    pair_interaction_bound = 0.5
    presence_loss_weight = 0.5
    monitor_per_set = 1
    monitor_seed = 6002
    monitor_nll_patience = 5
    monitor_nll_rise = 0.05
    batch_size = 7
    device = 'cpu'
    lr = 0.0003
    weight_decay = 0.0001
    warmup_epochs = 3
    num_epochs = 1
    lr_min = 1e-6
    ema_decay = 0.999
    use_amp = False
    clip_grad_norm = 5.
    ortho_loss_weight = 0.1
    early_stop_patience = 20
    architecture_name = 'tiny-A2-test'


class MemoryData(Dataset):
    def __init__(self, root, split):
        self.split = split
        self.is_train = split == 'Train'
        self.waveform_augment = object()
        self.spec_augment = object()
        self.samples = [(str(Path(root)/str(i)/split/'mix'/'fake.wav'), y.tolist(), int(y.sum()))
                        for i, y in enumerate(LEGAL_SET_MULTI_HOT)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        gen = torch.Generator().manual_seed(i)
        return (torch.randn(2,65,33,generator=gen), torch.randn(1,32,generator=gen),
                torch.tensor(self.samples[i][1]), self.samples[i][2])


class TestSharedEvidence(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(42)
        return MUARTV6A2(TinyA2())

    def inputs(self, n=7):
        return torch.randn(n,2,65,33), torch.randn(n,1,32)

    def test_additive_pairs_at_initialization_and_normalization(self):
        model = self.make_model().eval()
        out = model(*self.inputs())
        expected = torch.stack([out['single_logits'][:,i]+out['single_logits'][:,j]
                                for i,j in PAIR_CLASS_INDICES],1)
        torch.testing.assert_close(out['pair_logits'], expected, rtol=0, atol=0)
        self.assertEqual(out['presence_logits'].data_ptr(), out['single_logits'].data_ptr())
        torch.testing.assert_close(out['set_log_probs'].exp().sum(1), torch.ones(7))

    def test_interaction_is_bounded(self):
        model = self.make_model().eval()
        with torch.no_grad():
            model.classifier.pair_scorer[-1].bias.fill_(100.)
        out = model(*self.inputs())
        self.assertLessEqual(float(out['pair_interaction'].abs().max().detach()), 0.5)

    def test_double_nll_trains_single_evidence_and_encoder(self):
        model = self.make_model()
        out = model(*self.inputs(3))
        loss = F.nll_loss(out['set_log_probs'], torch.tensor([4,5,6]))
        loss.backward()
        self.assertGreater(float(model.classifier.single_scorer[-1].weight.grad.abs().sum()), 0.)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.backbone.parameters()
                               if p.grad is not None), 0.)

    def test_presence_supervises_noise_single_and_double(self):
        logits = torch.zeros(7,3,requires_grad=True)
        F.binary_cross_entropy_with_logits(logits, LEGAL_SET_MULTI_HOT).backward()
        # Every class of every row receives its correct positive/negative sign.
        self.assertTrue(torch.equal(logits.grad.sign(), 1-2*LEGAL_SET_MULTI_HOT))
        out = self.make_model()(*self.inputs())
        loss,nll,bce,_ = training.loss_components(out,LEGAL_SET_MULTI_HOT)
        torch.testing.assert_close(loss,nll+0.5*bce)

    def test_initial_shared_encoder_preserved_and_pair_capacity_reduced(self):
        torch.manual_seed(42)
        original = MUARTV6A(ConfigV6A2())
        torch.manual_seed(42)
        new = MUARTV6A2(ConfigV6A2())
        for prefix in ('backbone','demon_encoder'):
            for key, value in getattr(original,prefix).state_dict().items():
                torch.testing.assert_close(value,getattr(new,prefix).state_dict()[key],rtol=0,atol=0)
        old_n = sum(p.numel() for p in original.classifier.pair_scorer.parameters())
        new_n = sum(p.numel() for p in new.classifier.pair_scorer.parameters())
        self.assertLess(new_n,old_n/4)

    def test_monitor_disables_augmentation_without_changing_train_or_rng(self):
        source = MemoryData('/fake','Train')
        with training.preserve_rng():
            monitor = training.make_train_monitor(source,TinyA2())
            first = next(iter(monitor))
        self.assertTrue(source.is_train)
        self.assertIsNotNone(source.waveform_augment)
        self.assertFalse(monitor.dataset.is_train)
        self.assertIsNone(monitor.dataset.waveform_augment)
        self.assertIsNone(monitor.dataset.spec_augment)
        state = torch.get_rng_state().clone()
        with training.preserve_rng():
            torch.rand(10)
            list(monitor)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        torch.testing.assert_close(first[0],next(iter(monitor))[0])

    def test_monitor_rejects_val(self):
        with self.assertRaises(ValueError):
            training.make_train_monitor(MemoryData('/fake','Val'),TinyA2())

    def test_overfit_alert_requires_persistent_divergence(self):
        rows = [dict(epoch=1,val_nll=0.4,train_eval_nll=0.4)]
        self.assertFalse(training.overfit_signal(rows,TinyA2())['suspected_overfit'])
        rows += [dict(epoch=i,val_nll=0.6,train_eval_nll=0.1) for i in range(2,7)]
        self.assertTrue(training.overfit_signal(rows,TinyA2())['suspected_overfit'])

    def test_one_epoch_driver_saves_reloads_and_never_requests_test(self):
        # Synthetic in-memory data exercises training, EMA, reports and reload,
        # not just shape checks; no real dataset is read.
        with tempfile.TemporaryDirectory() as temp:
            cfg = TinyA2()
            cfg.data_combined = temp
            cfg.output_dir = str(Path(temp)/'run')
            trds, vads = MemoryData(temp,'Train'), MemoryData(temp,'Val')
            tr = DataLoader(trds,batch_size=7,collate_fn=collate_multi)
            va = DataLoader(vads,batch_size=7,collate_fn=collate_multi)
            snapshot = dict(resolved_config=collect_resolved_config(cfg), normalization_stats={}, dataset_manifest={})
            baseline_metrics = training.compute_metrics(
                F.one_hot(torch.arange(7),7).float(), LEGAL_SET_MULTI_HOT,
                LEGAL_SET_MULTI_HOT.sum(1).long(),cfg.class_names)
            baseline = dict(reloaded_val_metrics=baseline_metrics)
            args = argparse.Namespace(smoke_test=False,reference_dir=temp,seed=42)
            with patch.object(training,'validate_v6a2_config'), \
                 patch.object(training,'create_loaders',return_value=(tr,va,None,dict(n_train=7,n_val=7,n_test=None))) as loaders, \
                 patch.object(training,'build_experiment_snapshot',return_value=snapshot), \
                 patch.object(training,'protocol_check',return_value=baseline):
                result = training.run(cfg,args)
                loaders.assert_called_once_with(cfg,include_test=False)
            self.assertTrue(result['checkpoint_reload_verified'])
            for name in ('best_model.pt','best_nll_model.pt','comparison.md','val_predictions.csv',
                         'history.csv','epoch_metrics.json','overfit_watch.json'):
                self.assertTrue((Path(cfg.output_dir)/name).is_file(),name)
            with self.assertRaises(FileExistsError):
                training._prepare_empty_output(Path(cfg.output_dir))


if __name__ == '__main__':
    unittest.main()

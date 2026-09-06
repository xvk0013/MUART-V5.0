import json
import tempfile
import unittest
from pathlib import Path

# Keep the same Windows scientific-runtime import order as the formal trainer.
import numpy as np  # noqa: F401 - intentional runtime preflight

from config_v6a2 import ConfigV6A2
from config_v6d1 import ConfigV6D1, validate_v6d1_config
from train_v6d1 import FROZEN_CONFIG_KEYS, load_d1_audit


class TestV6D1(unittest.TestCase):
    def test_model_and_training_contract_matches_a2(self):
        a2, d1 = ConfigV6A2(), ConfigV6D1()
        validate_v6d1_config(d1)
        for key in FROZEN_CONFIG_KEYS:
            self.assertEqual(getattr(a2, key), getattr(d1, key), key)
        self.assertEqual(d1.experiment_stage, "V6-D1")

    def test_audit_gate_requires_improvement_and_frozen_val(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            good = {
                "status": "PASS", "failures": [],
                "val_content_hash": {"reference": {"sha256": "x"},
                                     "d1": {"sha256": "x"}},
                "recording_exposure": {
                    name: {"template_uniform_reference": {"cv": 1.0},
                           "d1_accepted": {"cv": 0.2, "coverage": 1.0}}
                    for name in ("Cargo", "Tanker", "Tug")}}
            (root / "d1_audit.json").write_text(json.dumps(good), encoding="utf-8")
            self.assertEqual(load_d1_audit(root)["status"], "PASS")
            good["recording_exposure"]["Cargo"]["d1_accepted"]["cv"] = 1.2
            (root / "d1_audit.json").write_text(json.dumps(good), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Cargo"):
                load_d1_audit(root)

if __name__ == "__main__":
    unittest.main(verbosity=2)

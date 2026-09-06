import tempfile
import unittest
from collections import Counter
from pathlib import Path

from audit_d1_dataset import (LABEL_FOLDER, audit_source_split_isolation,
                              dispersion, split_tree_content_hash)


class TestD1Audit(unittest.TestCase):
    def make_split(self, root: Path, payload: bytes = b"same"):
        for type_name in ("noise", "0", "1", "2", "0_1", "0_2", "1_2"):
            path = root / type_name / "Val" / "mix"
            path.mkdir(parents=True)
            (path / "combined_00001.wav").write_bytes(payload)

    def test_val_content_hash_is_path_and_byte_sensitive(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            left, right = root / "left", root / "right"
            self.make_split(left)
            self.make_split(right)
            self.assertEqual(split_tree_content_hash(left, "Val"),
                             split_tree_content_hash(right, "Val"))
            (right / "0" / "Val" / "mix" / "combined_00001.wav").write_bytes(b"changed")
            self.assertNotEqual(split_tree_content_hash(left, "Val")["sha256"],
                                split_tree_content_hash(right, "Val")["sha256"])

    def test_dispersion_detects_recording_balance(self):
        universe = {"a", "b", "c"}
        balanced = dispersion(Counter(a=10, b=10, c=10), universe)
        skewed = dispersion(Counter(a=28, b=1, c=1), universe)
        self.assertEqual(balanced["coverage"], 1.0)
        self.assertEqual(balanced["cv"], 0.0)
        self.assertGreater(skewed["cv"], balanced["cv"])
        self.assertGreater(skewed["gini"], balanced["gini"])

    def test_source_split_overlap_is_explicit(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            mappings = {}
            for label, folder in LABEL_FOLDER.items():
                mappings[label] = {
                    "train.wav": f"train_parent_{label}/recording",
                    "val.wav": f"val_parent_{label}/recording",
                }
                for split, filename in (("Train", "train.wav"), ("Val", "val.wav")):
                    directory = root / split / folder
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / filename).write_bytes(b"")
            isolated, _ = audit_source_split_isolation(root, mappings)
            self.assertTrue(all(item["recording_overlap"] == 0
                                for item in isolated.values()))
            (root / "Val" / LABEL_FOLDER[0] / "train.wav").write_bytes(b"")
            overlapped, failures = audit_source_split_isolation(root, mappings)
            self.assertEqual(overlapped["Cargo"]["template_overlap"], 1)
            self.assertEqual(overlapped["Cargo"]["recording_overlap"], 1)
            self.assertTrue(any("Cargo Train/Val recording overlap" in item
                                for item in failures))


if __name__ == "__main__":
    unittest.main(verbosity=2)

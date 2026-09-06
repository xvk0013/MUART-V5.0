import unittest

from d1_verdict import d1_diagnostic_verdict


def metrics(cargo_correct, single_swaps, tug_pair_swaps, overall_correct):
    order = ["noise", "Tanker", "Cargo", "Tug", "Tanker+Cargo",
             "Tanker+Tug", "Cargo+Tug"]
    matrix = [[0] * 7 for _ in range(7)]
    diagonal = [324, 180, cargo_correct, 220, 400, 380, 0]
    diagonal[-1] = overall_correct - sum(diagonal[:-1])
    for idx, value in enumerate(diagonal):
        matrix[idx][idx] = value
    matrix[2][1] = single_swaps
    matrix[6][5] = tug_pair_swaps
    return {"confusion_order": order, "confusion_matrix": matrix,
            "by_legal_set": {"Cargo": {"correct": cargo_correct}}}


class TestD1Verdict(unittest.TestCase):
    def test_strong_breakthrough(self):
        previous = metrics(147, 87, 222, 2005)
        current = metrics(155, 76, 205, 2006)
        verdict = d1_diagnostic_verdict(current, previous)
        self.assertTrue(verdict["diagnostic_success"])
        self.assertTrue(verdict["strong_breakthrough"])
        self.assertEqual(verdict["label"], "strong_breakthrough")

    def test_cargo_gain_with_identity_tradeoff_is_partial(self):
        previous = metrics(147, 87, 222, 2005)
        current = metrics(150, 90, 220, 2006)
        verdict = d1_diagnostic_verdict(current, previous)
        self.assertFalse(verdict["diagnostic_success"])
        self.assertEqual(verdict["label"], "partial_cargo_gain_with_tradeoff")


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest

from evaluate_snr import summarize_results


class EvaluateSnrSummaryTest(unittest.TestCase):
    def test_summary_uses_requested_low_snr_levels(self):
        results = [
            [
                {"snr": -10, "acc": 10.0, "macro_f1": 9.0},
                {"snr": -5, "acc": 50.0, "macro_f1": 49.0},
                {"snr": 0, "acc": 85.0, "macro_f1": 84.0},
                {"snr": 5, "acc": 90.0, "macro_f1": 89.0},
            ],
            [
                {"snr": -10, "acc": 12.0, "macro_f1": 11.0},
                {"snr": -5, "acc": 52.0, "macro_f1": 51.0},
                {"snr": 0, "acc": 87.0, "macro_f1": 86.0},
                {"snr": 5, "acc": 92.0, "macro_f1": 91.0},
            ],
        ]
        summary = summarize_results(results, [-10, -5, 0, 5])
        self.assertAlmostEqual(summary["by_snr"][0]["acc"]["mean"], 11.0)
        self.assertAlmostEqual(summary["low_snr_mean"]["acc"], 49.3333333333)
        self.assertEqual(summary["low_snr_mean"]["levels"], [-10.0, -5.0, 0.0])
        self.assertAlmostEqual(summary["all_snr_mean"]["acc"], 59.75)


if __name__ == "__main__":
    unittest.main()

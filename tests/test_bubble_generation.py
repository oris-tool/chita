import json
import random
import unittest
from pathlib import Path

import numpy as np

import dataset_graph as dg


class BubbleGenerationRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures_root = Path(__file__).resolve().parent / "fixtures" / "bubble_generation"
        cls.manifest = json.loads((fixtures_root / "manifest.json").read_text(encoding="utf-8"))

    def test_refactored_generator_matches_controlled_reference_fixtures(self):
        for case in self.manifest:
            with self.subTest(seed=case["seed"], total_internal_contacts=case["total_internal_contacts"]):
                random.seed(case["seed"])
                np.random.seed(case["seed"])
                result = dg.simulate_external_introduction(
                    n_nodes=case["n_nodes"],
                    total_internal_contacts=case["total_internal_contacts"],
                    tmax_after_intro=case["tmax_after_intro"],
                    effective_external_contacts=case["effective_external_contacts"],
                    seed=case["seed"],
                )
                payload = dg.build_dataset_event_sequence(result)
                expected = json.loads(Path(case["fixture_path"]).read_text(encoding="utf-8"))
                self.assertEqual(payload, expected)


if __name__ == "__main__":
    unittest.main()

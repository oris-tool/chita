import unittest
from collections import Counter

import scale_free_dataset_graph as sfdg


class ScaleFreeDatasetGenerationTest(unittest.TestCase):
    def test_scale_free_dataset_counts_match_requested_profile(self):
        for requested_internal_contacts in (32, 2500, 5000, 10000):
            with self.subTest(total_internal_contacts=requested_internal_contacts):
                result = sfdg.simulate_scale_free_introduction(
                    n_nodes=100,
                    total_external_contacts=1000,
                    total_symptom_observations=1000,
                    total_test_observations=1000,
                    total_internal_contacts=requested_internal_contacts,
                    effective_external_contacts=15,
                    barabasi_m=3,
                    seed=30,
                )
                payload = sfdg.build_dataset_event_sequence(result)
                event_counts = Counter(event["type"] for event in payload["events"])

                self.assertEqual(payload["n_subjects"], 100)
                self.assertEqual(payload["time_limit"], 84)
                self.assertEqual(payload["n_contacts"], requested_internal_contacts)
                self.assertEqual(event_counts["External"], 1000)
                self.assertEqual(event_counts["Symptoms"], 1000)
                self.assertEqual(event_counts["Test"], 1000)
                self.assertEqual(event_counts["Internal"], requested_internal_contacts)


if __name__ == "__main__":
    unittest.main()

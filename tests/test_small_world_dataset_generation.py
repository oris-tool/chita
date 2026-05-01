import unittest
from collections import Counter

import small_world_dataset_graph as swdg


class SmallWorldDatasetGenerationTest(unittest.TestCase):
    def test_small_world_dataset_counts_match_requested_profile(self):
        for requested_internal_contacts in (1800, 3600, 7200):
            with self.subTest(total_internal_contacts=requested_internal_contacts):
                result = swdg.simulate_small_world_introduction(
                    n_nodes=100,
                    total_external_contacts=1000,
                    total_symptom_observations=1000,
                    total_test_observations=1000,
                    total_internal_contacts=requested_internal_contacts,
                    effective_external_contacts=15,
                    watts_k=5,
                    rewire_probability=0.1,
                    seed=30,
                )
                payload = swdg.build_dataset_event_sequence(result)
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

import unittest

from probe.instruction_rewrite.online_vlm import _parse_online_output


class OnlineVlmOutputTest(unittest.TestCase):
    source = "put both the alphabet soup and the tomato sauce in the basket"

    def test_accepts_grounded_short_term_goal(self):
        result = _parse_online_output(
            '{"short_term_goal":"Put the alphabet soup in the basket.",'
            '"source_spans":["alphabet soup","basket"],"uncertain":false}',
            self.source,
        )

        self.assertTrue(result["rewrite_accepted"])
        self.assertIn("Overall task:", result["instruction"])
        self.assertIn("Immediate next step:", result["instruction"])

    def test_does_not_apply_lexical_safety_heuristics(self):
        result = _parse_online_output(
            '{"short_term_goal":"Put the orange juice can and the milk carton into the basket.",'
            '"source_spans":["basket"],"uncertain":false}',
            self.source,
        )

        self.assertTrue(result["rewrite_accepted"])
        self.assertIn("orange juice can", result["instruction"])
        self.assertIsNone(result["rejection_reason"])

    def test_preserves_uncertain_flag_without_safety_rejection(self):
        result = _parse_online_output(
            '{"short_term_goal":"Put the alphabet soup in the basket.",'
            '"source_spans":["alphabet soup","basket"],"uncertain":true}',
            self.source,
        )

        self.assertTrue(result["rewrite_accepted"])
        self.assertTrue(result["uncertain"])

    def test_rejects_unstructured_output(self):
        result = _parse_online_output("Put the orange juice into the basket.", self.source)

        self.assertFalse(result["rewrite_accepted"])
        self.assertEqual(result["instruction"], self.source)
        self.assertEqual(result["rejection_reason"], "unstructured_output")


if __name__ == "__main__":
    unittest.main()

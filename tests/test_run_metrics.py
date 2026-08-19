from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from newme.portal import (
    _human_label_instances,
    _labels_match,
    _llm_hit_instances,
    _match_hits_to_human_instances,
    _match_wmn_groups_to_sequences,
    _match_wmn_label_instances,
    _quote_word_overlap,
    _ranges_overlap,
)


class RangesOverlapTests(unittest.TestCase):
    def test_identical_single_utterance_overlaps(self) -> None:
        self.assertTrue(_ranges_overlap(5, 5, 5, 5))

    def test_disjoint_ranges_do_not_overlap(self) -> None:
        self.assertFalse(_ranges_overlap(1, 2, 3, 4))

    def test_partially_overlapping_ranges_overlap(self) -> None:
        self.assertTrue(_ranges_overlap(1, 5, 5, 9))


class InstanceExtractionTests(unittest.TestCase):
    def test_human_label_instances_skips_invalid_labels(self) -> None:
        seq = SimpleNamespace(wmn_id="seq-A", labels=[
            SimpleNamespace(name="Indicator", start_index=2, end_index=2),
            SimpleNamespace(name="NotALabel", start_index=3, end_index=3),
            SimpleNamespace(name="Trigger", start_index=None, end_index=1),
        ])

        instances = _human_label_instances([seq])

        self.assertEqual(instances, [("Indicator", 2, 2)])

    def test_llm_hit_instances_keeps_hits_with_unrecognized_labels(self) -> None:
        result = SimpleNamespace(output=[
            {"label": "Indicator", "utterance_start_index": 1, "utterance_end_index": 1},
            {"label": "Bogus", "utterance_start_index": 2, "utterance_end_index": 2},
            {"utterance_start_index": 3, "utterance_end_index": 3},
        ])

        instances = _llm_hit_instances(result)

        self.assertEqual(
            instances,
            [("Indicator", 1, 1), ("Bogus", 2, 2), (None, 3, 3)],
        )


class MatchHitsToHumanInstancesTests(unittest.TestCase):
    def test_same_label_same_utterance_is_a_true_positive_regardless_of_quote_position(self) -> None:
        # Rule 1: position of the quote within the utterance never enters the
        # comparison — matching is purely (label, utterance range).
        human = [("Indicator", 5, 5)]
        llm = [("Indicator", 5, 5)]

        match = _match_hits_to_human_instances(human, llm)

        self.assertEqual(match, {"hit_tp": 1, "hit_fp": 0, "human_tp": 1, "human_fn": 0})

    def test_multi_utterance_hit_matches_if_any_spanned_utterance_has_same_label(self) -> None:
        # Rule 2: the human annotation only covers utterance 6 out of the hit's
        # 5-7 span; that's still enough for a true positive.
        human = [("Negotiation", 6, 6)]
        llm = [("Negotiation", 5, 7)]

        match = _match_hits_to_human_instances(human, llm)

        self.assertEqual(match, {"hit_tp": 1, "hit_fp": 0, "human_tp": 1, "human_fn": 0})

    def test_different_label_on_same_utterance_does_not_match(self) -> None:
        human = [("Trigger", 5, 5)]
        llm = [("Indicator", 5, 5)]

        match = _match_hits_to_human_instances(human, llm)

        self.assertEqual(match, {"hit_tp": 0, "hit_fp": 1, "human_tp": 0, "human_fn": 1})

    def test_non_overlapping_ranges_do_not_match(self) -> None:
        human = [("Indicator", 1, 1)]
        llm = [("Indicator", 9, 9)]

        match = _match_hits_to_human_instances(human, llm)

        self.assertEqual(match, {"hit_tp": 0, "hit_fp": 1, "human_tp": 0, "human_fn": 1})

    def test_one_hit_can_recall_multiple_human_instances(self) -> None:
        human = [("Indicator", 5, 5), ("Indicator", 7, 7)]
        llm = [("Indicator", 5, 7)]

        match = _match_hits_to_human_instances(human, llm)

        self.assertEqual(match, {"hit_tp": 1, "hit_fp": 0, "human_tp": 2, "human_fn": 0})

    def test_multiple_hits_can_recall_the_same_human_instance(self) -> None:
        human = [("Indicator", 5, 5)]
        llm = [("Indicator", 5, 5), ("Indicator", 4, 6)]

        match = _match_hits_to_human_instances(human, llm)

        self.assertEqual(match, {"hit_tp": 2, "hit_fp": 0, "human_tp": 1, "human_fn": 0})


class MatchWmnGroupsToSequencesTests(unittest.TestCase):
    def test_single_overlapping_group_and_sequence_are_paired(self) -> None:
        human = [("Indicator", 5, 5, "seq-A"), ("Trigger", 2, 2, "seq-A")]
        llm = [("Indicator", 5, 5, 1), ("Trigger", 2, 2, 1)]

        match = _match_wmn_groups_to_sequences(human, llm)

        self.assertEqual(match["group_to_sequence"], {1: "seq-A"})
        self.assertEqual(match["sequence_to_group"], {"seq-A": 1})
        self.assertEqual(match["unmatched_groups"], [])
        self.assertEqual(match["unmatched_sequences"], [])

    def test_group_with_no_overlapping_sequence_is_unmatched(self) -> None:
        human = [("Indicator", 5, 5, "seq-A")]
        llm = [("Indicator", 99, 99, 1)]

        match = _match_wmn_groups_to_sequences(human, llm)

        self.assertEqual(match["group_to_sequence"], {})
        self.assertEqual(match["unmatched_groups"], [1])
        self.assertEqual(match["unmatched_sequences"], ["seq-A"])

    def test_sequence_with_no_overlapping_group_is_unmatched(self) -> None:
        human = [("Indicator", 5, 5, "seq-A"), ("Indicator", 99, 99, "seq-B")]
        llm = [("Indicator", 5, 5, 1)]

        match = _match_wmn_groups_to_sequences(human, llm)

        self.assertEqual(match["group_to_sequence"], {1: "seq-A"})
        self.assertEqual(match["unmatched_groups"], [])
        self.assertEqual(match["unmatched_sequences"], ["seq-B"])

    def test_only_indicator_overlap_decides_the_pairing(self) -> None:
        # Group 1's Trigger/Negotiation overlap seq-A, but its Indicator doesn't —
        # under the Indicator-only rule that's not a match at all.
        human = [
            ("Trigger", 1, 1, "seq-A"),
            ("Indicator", 5, 5, "seq-A"),
            ("Negotiation", 6, 6, "seq-A"),
        ]
        llm = [
            ("Trigger", 1, 1, 1),
            ("Indicator", 99, 99, 1),
            ("Negotiation", 6, 6, 1),
        ]

        match = _match_wmn_groups_to_sequences(human, llm)

        self.assertEqual(match["group_to_sequence"], {})
        self.assertEqual(match["unmatched_groups"], [1])
        self.assertEqual(match["unmatched_sequences"], ["seq-A"])

    def test_trigger_and_negotiation_overlap_does_not_substitute_for_indicator_overlap(self) -> None:
        # Same as above but confirms it holds even when every other label matches.
        human = [("Indicator", 5, 5, "seq-A")]
        llm = [("Indicator", 9, 9, 1)]

        match = _match_wmn_groups_to_sequences(human, llm)

        self.assertEqual(match["group_to_sequence"], {})

    def test_two_groups_overlapping_one_sequence_indicator_ties_broken_deterministically(self) -> None:
        # Both group 1 and group 2's Indicators overlap seq-A's Indicator — tied
        # at one match each, so the lower-numbered group wins the pairing.
        human = [("Indicator", 5, 5, "seq-A")]
        llm = [("Indicator", 5, 5, 1), ("Indicator", 4, 6, 2)]

        match = _match_wmn_groups_to_sequences(human, llm)

        self.assertEqual(match["group_to_sequence"], {1: "seq-A"})
        self.assertEqual(match["unmatched_groups"], [2])
        self.assertEqual(match["unmatched_sequences"], [])

    def test_one_group_overlapping_two_sequence_indicators_ties_broken_deterministically(self) -> None:
        human = [("Indicator", 5, 5, "seq-A"), ("Indicator", 4, 6, "seq-B")]
        llm = [("Indicator", 5, 5, 1)]

        match = _match_wmn_groups_to_sequences(human, llm)

        self.assertEqual(match["group_to_sequence"], {1: "seq-A"})
        self.assertEqual(match["unmatched_sequences"], ["seq-B"])

    def test_hits_and_labels_without_a_group_or_sequence_are_ignored(self) -> None:
        human = [("Indicator", 5, 5, None)]
        llm = [("Indicator", 5, 5, None)]

        match = _match_wmn_groups_to_sequences(human, llm)

        self.assertEqual(match["group_to_sequence"], {})
        self.assertEqual(match["sequence_to_group"], {})
        self.assertEqual(match["unmatched_groups"], [])
        self.assertEqual(match["unmatched_sequences"], [])


class QuoteWordOverlapTests(unittest.TestCase):
    def test_shared_word_is_a_match_regardless_of_case(self) -> None:
        self.assertTrue(_quote_word_overlap("Free work", "WORK"))

    def test_shared_word_is_a_match_even_at_different_positions_in_the_phrase(self) -> None:
        self.assertTrue(_quote_word_overlap("the concept of work", "work is valuable"))

    def test_no_shared_word_is_not_a_match(self) -> None:
        self.assertFalse(_quote_word_overlap("free market", "minimum wage"))

    def test_empty_quotes_never_match(self) -> None:
        self.assertFalse(_quote_word_overlap("", "work"))
        self.assertFalse(_quote_word_overlap("work", ""))


class LabelsMatchTests(unittest.TestCase):
    def test_trigger_matches_on_word_regardless_of_utterance_position(self) -> None:
        llm_items = [{"start": 6, "end": 6, "quote": 'Eliminating "Free work"'}]
        human_items = [{"start": 40, "end": 40, "quote": "the nature of work"}]

        self.assertTrue(_labels_match("Trigger", llm_items, human_items))

    def test_trigger_with_no_shared_word_does_not_match(self) -> None:
        llm_items = [{"start": 6, "end": 6, "quote": "minimum wage"}]
        human_items = [{"start": 6, "end": 6, "quote": "free market"}]

        self.assertFalse(_labels_match("Trigger", llm_items, human_items))

    def test_negotiation_matches_on_any_overlap_ignoring_quote(self) -> None:
        llm_items = [{"start": 8, "end": 9, "quote": "completely different text"}]
        human_items = [{"start": 9, "end": 9, "quote": "also completely different"}]

        self.assertTrue(_labels_match("Negotiation", llm_items, human_items))

    def test_negotiation_with_no_overlap_does_not_match(self) -> None:
        llm_items = [{"start": 8, "end": 8, "quote": "x"}]
        human_items = [{"start": 20, "end": 20, "quote": "x"}]

        self.assertFalse(_labels_match("Negotiation", llm_items, human_items))

    def test_one_matching_pair_among_several_human_items_is_enough(self) -> None:
        # Mirrors the real data: a human annotator sometimes marks the same
        # Trigger word at several points; only one needs to match.
        llm_items = [{"start": 6, "end": 6, "quote": "work"}]
        human_items = [
            {"start": 1, "end": 1, "quote": "free market"},
            {"start": 18, "end": 18, "quote": "work"},
            {"start": 25, "end": 25, "quote": "minimum wage"},
        ]

        self.assertTrue(_labels_match("Trigger", llm_items, human_items))


class MatchWmnLabelInstancesTests(unittest.TestCase):
    def test_trigger_indicator_negotiation_all_match_within_a_paired_wmn(self) -> None:
        human = [
            {"name": "Indicator", "start": 5, "end": 5, "sequence": "seq-A", "quote": "what is work?"},
            {"name": "Trigger", "start": 1, "end": 1, "sequence": "seq-A", "quote": "free work"},
            {"name": "Negotiation", "start": 6, "end": 6, "sequence": "seq-A", "quote": "anything"},
        ]
        llm = [
            {"name": "Indicator", "start": 5, "end": 5, "group": 1, "quote": "what is work?"},
            # Trigger anchored to a different utterance than the human's, but shares the word "work".
            {"name": "Trigger", "start": 9, "end": 9, "group": 1, "quote": "the concept of work"},
            {"name": "Negotiation", "start": 6, "end": 6, "group": 1, "quote": "something else"},
        ]

        match = _match_wmn_label_instances(human, llm)

        self.assertEqual(match, {"hit_tp": 3, "hit_fp": 0, "human_tp": 3, "human_fn": 0})

    def test_multiple_human_trigger_rows_for_one_wmn_count_as_a_single_slot(self) -> None:
        human = [
            {"name": "Indicator", "start": 5, "end": 5, "sequence": "seq-A", "quote": "what is work?"},
            {"name": "Trigger", "start": 1, "end": 1, "sequence": "seq-A", "quote": "work"},
            {"name": "Trigger", "start": 3, "end": 3, "sequence": "seq-A", "quote": "work"},
            {"name": "Trigger", "start": 18, "end": 18, "sequence": "seq-A", "quote": "work"},
        ]
        llm = [
            {"name": "Indicator", "start": 5, "end": 5, "group": 1, "quote": "what is work?"},
            {"name": "Trigger", "start": 9, "end": 9, "group": 1, "quote": "work"},
        ]

        match = _match_wmn_label_instances(human, llm)

        # 2 slots total (Indicator + Trigger), not 4 — the 3 human Trigger rows
        # collapse into 1.
        self.assertEqual(match["human_tp"], 2)
        self.assertEqual(match["human_fn"], 0)
        self.assertEqual(match["hit_tp"], 2)
        self.assertEqual(match["hit_fp"], 0)

    def test_trigger_word_mismatch_within_a_paired_wmn_is_fp_and_fn(self) -> None:
        human = [
            {"name": "Indicator", "start": 5, "end": 5, "sequence": "seq-A", "quote": "what is work?"},
            {"name": "Trigger", "start": 1, "end": 1, "sequence": "seq-A", "quote": "free market"},
        ]
        llm = [
            {"name": "Indicator", "start": 5, "end": 5, "group": 1, "quote": "what is work?"},
            {"name": "Trigger", "start": 1, "end": 1, "group": 1, "quote": "minimum wage"},
        ]

        match = _match_wmn_label_instances(human, llm)

        self.assertEqual(match, {"hit_tp": 1, "hit_fp": 1, "human_tp": 1, "human_fn": 1})

    def test_unpaired_group_is_a_false_positive_on_every_label_it_has(self) -> None:
        human = [
            {"name": "Indicator", "start": 5, "end": 5, "sequence": "seq-A", "quote": "what is work?"},
        ]
        llm = [
            # Indicator doesn't overlap seq-A's, so the group never pairs.
            {"name": "Indicator", "start": 99, "end": 99, "group": 1, "quote": "unrelated"},
            {"name": "Trigger", "start": 1, "end": 1, "group": 1, "quote": "free market"},
        ]

        match = _match_wmn_label_instances(human, llm)

        self.assertEqual(match["hit_tp"], 0)
        self.assertEqual(match["hit_fp"], 2)  # Indicator + Trigger, both unmatched

    def test_unpaired_sequence_is_a_false_negative_on_every_label_it_has(self) -> None:
        human = [
            {"name": "Indicator", "start": 5, "end": 5, "sequence": "seq-A", "quote": "what is work?"},
            {"name": "Negotiation", "start": 6, "end": 6, "sequence": "seq-A", "quote": "anything"},
        ]
        llm: list[dict] = []

        match = _match_wmn_label_instances(human, llm)

        self.assertEqual(match["human_tp"], 0)
        self.assertEqual(match["human_fn"], 2)  # Indicator + Negotiation, both missed


if __name__ == "__main__":
    unittest.main()

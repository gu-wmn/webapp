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
    _annotate_dialogue_utterances,
    _group_label_links_by_wmn,
    _result_label_payload,
)


class GroupLabelLinksByWmnTests(unittest.TestCase):
    def test_no_wmn_group_yields_a_single_ungrouped_bucket(self) -> None:
        links = [
            {"name": "Indicator", "excerpt": "a"},
            {"name": "Trigger", "excerpt": "b"},
        ]

        groups = _group_label_links_by_wmn(links)

        self.assertEqual(groups, [{"group": None, "wmn_type": "", "links": links}])

    def test_distinct_wmn_groups_are_split_in_first_seen_order(self) -> None:
        links = [
            {"name": "Trigger", "excerpt": "a", "wmn_group": 2},
            {"name": "Indicator", "excerpt": "b", "wmn_group": 1},
            {"name": "Negotiation", "excerpt": "c", "wmn_group": 2},
            {"name": "Indicator", "excerpt": "d", "wmn_group": 1},
        ]

        groups = _group_label_links_by_wmn(links)

        self.assertEqual([g["group"] for g in groups], [2, 1])
        self.assertEqual([l["excerpt"] for l in groups[0]["links"]], ["a", "c"])
        self.assertEqual([l["excerpt"] for l in groups[1]["links"]], ["b", "d"])

    def test_links_without_a_group_land_in_a_trailing_bucket(self) -> None:
        links = [
            {"name": "Indicator", "excerpt": "a", "wmn_group": 1},
            {"name": "Indicator", "excerpt": "b", "wmn_group": None},
        ]

        groups = _group_label_links_by_wmn(links)

        self.assertEqual([g["group"] for g in groups], [1, None])
        self.assertEqual([l["excerpt"] for l in groups[1]["links"]], ["b"])

    def test_group_type_is_taken_from_the_indicator_hit(self) -> None:
        links = [
            {"name": "Trigger", "excerpt": "a", "wmn_group": 1, "wmn_type": "other"},
            {"name": "Indicator", "excerpt": "b", "wmn_group": 1, "wmn_type": "non-understanding"},
            {"name": "Negotiation", "excerpt": "c", "wmn_group": 1, "wmn_type": "disagreement"},
        ]

        groups = _group_label_links_by_wmn(links)

        self.assertEqual(groups[0]["wmn_type"], "non-understanding")
        self.assertTrue(all(l["wmn_type"] == "non-understanding" for l in groups[0]["links"]))

    def test_group_type_falls_back_to_any_hit_when_indicator_has_none(self) -> None:
        links = [
            {"name": "Trigger", "excerpt": "a", "wmn_group": 1, "wmn_type": "other"},
            {"name": "Indicator", "excerpt": "b", "wmn_group": 1, "wmn_type": ""},
        ]

        groups = _group_label_links_by_wmn(links)

        self.assertEqual(groups[0]["wmn_type"], "other")
        self.assertTrue(all(l["wmn_type"] == "other" for l in groups[0]["links"]))


class WmnGroupPassthroughTests(unittest.TestCase):
    def test_result_label_payload_carries_wmn_group_from_hit(self) -> None:
        utterances = [
            {"author": "A", "text": "Define free market."},
            {"author": "B", "text": "That's a fair question."},
        ]
        result = SimpleNamespace(output=[
            {
                "label": "Indicator",
                "utterance_start_index": 0,
                "utterance_end_index": 0,
                "quote": "Define free market.",
                "wmn_group": 3,
            },
        ])

        labels = _result_label_payload(result, utterances)

        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0]["wmn_group"], 3)

    def test_annotate_dialogue_utterances_carries_wmn_group_into_label_links(self) -> None:
        utterances = [{"author": "A", "text": "Define free market."}]
        labels = [{
            "name": "Indicator",
            "utterance_start_index": 0,
            "utterance_end_index": 0,
            "char_start_index": 0,
            "char_end_index": 6,
            "quote": "Define",
            "wmn_group": 5,
        }]

        _annotated, label_links = _annotate_dialogue_utterances(
            utterances, labels, anchor_prefix="llm-label"
        )

        self.assertEqual(len(label_links), 1)
        self.assertEqual(label_links[0]["wmn_group"], 5)


if __name__ == "__main__":
    unittest.main()

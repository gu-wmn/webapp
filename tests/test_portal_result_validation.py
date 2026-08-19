from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from newme.portal import _derive_spans_from_quote, _validate_result_hits
from newme.models.experiment import MULTI_UTTERANCE_QUOTE_SEPARATOR


class ResultValidationTests(unittest.TestCase):
    def test_simplified_output_accepts_quote_within_spanned_utterance(self) -> None:
        utterances = [{"author": "A", "text": "Well, what do you mean by that exactly?"}]
        result_output = [{
            "label": "Indicator",
            "utterance_start_index": 0,
            "utterance_end_index": 0,
            "quote": "what do you mean",
        }]

        issues = _validate_result_hits(result_output, utterances, output_format="simplified")

        self.assertEqual(issues, [])

    def test_multi_utterance_hit_accepts_full_verbatim_quote(self) -> None:
        utterances = [
            {"author": "A", "text": "Define free market."},
            {"author": "B", "text": "That's a fair question."},
            {"author": "A", "text": "You still haven't defined what you mean."},
        ]
        result_output = [{
            "label": "Negotiation",
            "utterance_start_index": 0,
            "utterance_end_index": 2,
            "quote": "Define free market.\nThat's a fair question.\nYou still haven't defined what you mean.",
        }]

        issues = _validate_result_hits(result_output, utterances, output_format="simplified")

        self.assertEqual(issues, [])

    def test_multi_utterance_hit_accepts_boundary_quote_format(self) -> None:
        utterances = [
            {"author": "A", "text": "Define free market."},
            {"author": "B", "text": "That's a fair question, let me think about it for a while."},
            {"author": "A", "text": "You still haven't defined what you mean."},
        ]
        quote = f"Define free market.{MULTI_UTTERANCE_QUOTE_SEPARATOR}You still haven't defined what you mean."
        result_output = [{
            "label": "Negotiation",
            "utterance_start_index": 0,
            "utterance_end_index": 2,
            "quote": quote,
        }]

        issues = _validate_result_hits(result_output, utterances, output_format="simplified")

        self.assertEqual(issues, [])

    def test_multi_utterance_hit_rejects_boundary_text_not_present(self) -> None:
        utterances = [
            {"author": "A", "text": "Define free market."},
            {"author": "B", "text": "That's a fair question."},
            {"author": "A", "text": "You still haven't defined what you mean."},
        ]
        quote = f"Something never said{MULTI_UTTERANCE_QUOTE_SEPARATOR}Also never said"
        result_output = [{
            "label": "Negotiation",
            "utterance_start_index": 0,
            "utterance_end_index": 2,
            "quote": quote,
        }]

        issues = _validate_result_hits(result_output, utterances, output_format="simplified")

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["severity"], "warning")

    def test_derive_spans_from_quote_locates_boundary_format(self) -> None:
        utterances = [
            {"author": "A", "text": "Define free market."},
            {"author": "B", "text": "That's a fair question."},
            {"author": "A", "text": "You still haven't defined what you mean."},
        ]
        quote = f"free market.{MULTI_UTTERANCE_QUOTE_SEPARATOR}haven't defined"

        spans = _derive_spans_from_quote(utterances, 0, 2, quote)

        self.assertEqual(len(spans), 1)
        utt_s, char_s, utt_e, char_e = spans[0]
        self.assertEqual(utt_s, 0)
        self.assertEqual(utt_e, 2)
        self.assertEqual(utterances[0]["text"][char_s:], "free market.")
        self.assertEqual(utterances[2]["text"][:char_e], "You still haven't defined")

    def test_detailed_output_requires_exact_quote_match(self) -> None:
        utterances = [{"author": "A", "text": "Well, what do you mean by that exactly?"}]
        result_output = [{
            "label": "Indicator",
            "utterance_start_index": 0,
            "utterance_end_index": 0,
            "char_start_index": 6,
            "char_end_index": 22,
            "quote": "what do you mean by",
        }]

        issues = _validate_result_hits(result_output, utterances, output_format="detailed")

        self.assertEqual(len(issues), 1)
        self.assertIn("does not exactly match", issues[0]["message"])


if __name__ == "__main__":
    unittest.main()

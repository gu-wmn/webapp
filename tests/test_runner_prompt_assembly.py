from __future__ import annotations

import sys
import unittest
import json
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from newme.runner import assemble_prompt_text, format_dialogue
from newme.models.experiment import (
    DIALOGUE_INPUT_INSTRUCTIONS_DEFAULT,
    GLOBAL_TEMPLATE_DEFAULT,
    PREVIOUS_OUTPUT_INSTRUCTIONS_DEFAULT,
    REGEX_INPUT_INSTRUCTIONS_DEFAULT,
)


def _user_settings(**overrides) -> SimpleNamespace:
    defaults = dict(
        effective_global_template="",
        effective_free_text_appendix="",
        effective_dialogue_input_instructions=DIALOGUE_INPUT_INSTRUCTIONS_DEFAULT,
        effective_regex_input_instructions=REGEX_INPUT_INSTRUCTIONS_DEFAULT,
        effective_previous_output_instructions=PREVIOUS_OUTPUT_INSTRUCTIONS_DEFAULT,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _prompt(**overrides) -> SimpleNamespace:
    defaults = dict(
        include_global_template=False,
        include_dialogue=True,
        include_regex_candidates=False,
        prompt_text="Prompt body",
        output_format=None,
        position=1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class PromptAssemblyTests(unittest.TestCase):
    def test_dialogue_is_serialized_as_json_payload(self) -> None:
        utterances = [
            SimpleNamespace(position=0, author="A", text="Hello"),
            SimpleNamespace(position=1, author="B", text="What do you mean?"),
        ]

        payload = json.loads(format_dialogue(utterances, "dlg-1", "switchboard-corpus"))

        self.assertEqual(payload["dialogue"]["dialogue_id"], "dlg-1")
        self.assertEqual(payload["dialogue"]["corpus_codename"], "switchboard-corpus")
        self.assertEqual(
            payload["dialogue"]["utterances"],
            [
                {"utterance_index": 0, "speaker": "A", "text": "Hello"},
                {"utterance_index": 1, "speaker": "B", "text": "What do you mean?"},
            ],
        )

    def test_global_prompt_template_is_included_when_enabled(self) -> None:
        prompt = _prompt(include_global_template=True)
        user_settings = _user_settings(effective_global_template="Global prompt template")

        assembled = assemble_prompt_text(prompt, "Dialogue text", "", user_settings)

        self.assertIn("Global prompt template", assembled)
        self.assertIn("Prompt body", assembled)
        self.assertIn("Dialogue text", assembled)

    def test_global_prompt_template_is_omitted_when_disabled(self) -> None:
        prompt = _prompt(include_global_template=False)
        user_settings = _user_settings(effective_global_template="Global prompt template")

        assembled = assemble_prompt_text(prompt, "Dialogue text", "", user_settings)

        self.assertNotIn("Global prompt template", assembled)
        self.assertIn("Prompt body", assembled)
        self.assertIn("Dialogue text", assembled)

    def test_default_global_prompt_template_is_used_when_no_custom_value_exists(self) -> None:
        prompt = _prompt(include_global_template=True)
        user_settings = _user_settings(effective_global_template=GLOBAL_TEMPLATE_DEFAULT)

        assembled = assemble_prompt_text(prompt, "Dialogue text", "", user_settings)

        self.assertIn(GLOBAL_TEMPLATE_DEFAULT, assembled)
        self.assertIn("Prompt body", assembled)
        self.assertIn("Dialogue text", assembled)

    def test_dialogue_is_always_included_for_the_first_prompt(self) -> None:
        prompt = _prompt(position=1, include_dialogue=False)
        user_settings = _user_settings()

        assembled = assemble_prompt_text(prompt, "Dialogue text", "", user_settings)

        self.assertIn("Dialogue text", assembled)

    def test_dialogue_input_can_be_disabled_after_first_prompt(self) -> None:
        enabled_prompt = _prompt(position=2, include_dialogue=True)
        disabled_prompt = _prompt(position=2, include_dialogue=False)
        user_settings = _user_settings()

        enabled_assembled = assemble_prompt_text(enabled_prompt, "Dialogue text", "", user_settings)
        disabled_assembled = assemble_prompt_text(disabled_prompt, "Dialogue text", "", user_settings)

        self.assertIn("Dialogue text", enabled_assembled)
        self.assertNotIn("Dialogue text", disabled_assembled)

    def test_previous_output_is_included_only_after_first_prompt(self) -> None:
        first_prompt = _prompt(position=1)
        later_prompt = _prompt(position=2)
        user_settings = _user_settings()

        first_assembled = assemble_prompt_text(first_prompt, "Dialogue text", "Previous text", user_settings)
        later_assembled = assemble_prompt_text(later_prompt, "Dialogue text", "Previous text", user_settings)

        self.assertNotIn("Previous text", first_assembled)
        self.assertIn("Previous text", later_assembled)

    def test_regex_candidates_are_included_only_when_the_prompt_opts_in(self) -> None:
        without_regex = _prompt(include_regex_candidates=False)
        with_regex = _prompt(include_regex_candidates=True)
        user_settings = _user_settings()

        without_assembled = assemble_prompt_text(
            without_regex, "Dialogue text", "", user_settings, "Regex hits"
        )
        with_assembled = assemble_prompt_text(
            with_regex, "Dialogue text", "", user_settings, "Regex hits"
        )

        self.assertNotIn("Regex hits", without_assembled)
        self.assertIn("Regex hits", with_assembled)

    def test_input_blocks_are_prefixed_with_schema_explanations(self) -> None:
        first_prompt = _prompt(position=1, include_regex_candidates=True)
        later_prompt = _prompt(position=2)
        user_settings = _user_settings()

        first_assembled = assemble_prompt_text(
            first_prompt, "Dialogue text", "", user_settings, "Regex hits"
        )
        self.assertIn(DIALOGUE_INPUT_INSTRUCTIONS_DEFAULT, first_assembled)
        self.assertIn(REGEX_INPUT_INSTRUCTIONS_DEFAULT, first_assembled)

        later_assembled = assemble_prompt_text(later_prompt, "Dialogue text", "Previous text", user_settings)
        self.assertIn(PREVIOUS_OUTPUT_INSTRUCTIONS_DEFAULT, later_assembled)

    def test_custom_input_instructions_from_settings_are_used(self) -> None:
        prompt = _prompt(position=1)
        user_settings = _user_settings(
            effective_dialogue_input_instructions="Custom dialogue instructions"
        )

        assembled = assemble_prompt_text(prompt, "Dialogue text", "", user_settings)

        self.assertIn("Custom dialogue instructions", assembled)
        self.assertNotIn(DIALOGUE_INPUT_INSTRUCTIONS_DEFAULT, assembled)


class AdaptiveNumCtxTests(unittest.TestCase):
    def test_short_prompt_rounds_up_to_cover_the_output_reserve(self) -> None:
        # A trivial prompt still needs enough room for the output reserve
        # alone (2048 tokens), which rounds up to the next power of two.
        from newme.runner import _adaptive_num_ctx

        self.assertEqual(_adaptive_num_ctx("short prompt"), 4096)

    def test_result_is_always_a_power_of_two(self) -> None:
        from newme.runner import _adaptive_num_ctx

        for length in (100, 5_000, 20_000, 100_000):
            num_ctx = _adaptive_num_ctx("x" * length)
            self.assertEqual(num_ctx & (num_ctx - 1), 0, f"{num_ctx} for length {length}")

    def test_result_covers_input_plus_output_reserve(self) -> None:
        from newme.runner import _adaptive_num_ctx, _CHARS_PER_TOKEN, _OUTPUT_TOKEN_RESERVE

        text = "x" * 40_000
        num_ctx = _adaptive_num_ctx(text)
        needed = len(text) / _CHARS_PER_TOKEN + _OUTPUT_TOKEN_RESERVE

        self.assertGreaterEqual(num_ctx, needed)
        # A power of two rounds up by less than double what's actually needed.
        self.assertLess(num_ctx / 2, needed)

    def test_longer_prompt_never_yields_a_smaller_window(self) -> None:
        from newme.runner import _adaptive_num_ctx

        self.assertLessEqual(_adaptive_num_ctx("x" * 1_000), _adaptive_num_ctx("x" * 50_000))

    def test_result_is_capped_at_the_maximum(self) -> None:
        from newme.runner import _adaptive_num_ctx, _MAX_NUM_CTX

        self.assertEqual(_adaptive_num_ctx("x" * 10_000_000), _MAX_NUM_CTX)


if __name__ == "__main__":
    unittest.main()

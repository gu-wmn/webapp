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

        self.assertEqual(_adaptive_num_ctx(len("short prompt")), 4096)

    def test_result_is_always_a_power_of_two(self) -> None:
        from newme.runner import _adaptive_num_ctx

        for length in (100, 5_000, 20_000, 100_000):
            num_ctx = _adaptive_num_ctx(length)
            self.assertEqual(num_ctx & (num_ctx - 1), 0, f"{num_ctx} for length {length}")

    def test_result_covers_input_plus_output_reserve_with_safety_margin(self) -> None:
        from newme.runner import (
            _adaptive_num_ctx,
            _CHARS_PER_TOKEN,
            _OUTPUT_TOKEN_RESERVE_FLOOR,
            _OUTPUT_TOKEN_RESERVE_FRACTION,
            _OUTPUT_TOKEN_RESERVE_CAP,
            _SAFETY_MARGIN_MULTIPLIER,
        )

        length = 40_000
        num_ctx = _adaptive_num_ctx(length)
        input_tokens = length / _CHARS_PER_TOKEN
        reserve = min(
            _OUTPUT_TOKEN_RESERVE_CAP,
            max(_OUTPUT_TOKEN_RESERVE_FLOOR, int(input_tokens * _OUTPUT_TOKEN_RESERVE_FRACTION)),
        )
        needed = input_tokens + reserve

        self.assertGreaterEqual(num_ctx, needed)

    def test_headroom_never_drops_below_the_safety_margin_anywhere_in_a_tier(self) -> None:
        # The whole point of padding before rounding: unlike bare
        # round-up-to-power-of-two, headroom shouldn't shrink toward ~0% right
        # before the next doubling — it should stay above a floor everywhere.
        from newme.runner import _adaptive_num_ctx, _CHARS_PER_TOKEN, _SAFETY_MARGIN_MULTIPLIER

        min_headroom_ratio = float("inf")
        for length in range(1_000, 200_000, 2_777):  # odd step to sample mid-tier points, not just edges
            num_ctx = _adaptive_num_ctx(length)
            input_tokens = length / _CHARS_PER_TOKEN
            min_headroom_ratio = min(min_headroom_ratio, num_ctx / input_tokens)

        # Some slack below the nominal 1.42x: the output reserve and the
        # ceil/floor rounding both eat a little into the pure input-vs-window ratio.
        self.assertGreater(min_headroom_ratio, _SAFETY_MARGIN_MULTIPLIER * 0.9)

    def test_there_is_no_upper_cap(self) -> None:
        # Correctness matters more than bounding worst-case memory use — a
        # large enough dialogue should keep getting a proportionally larger
        # window rather than being clamped.
        from newme.runner import _adaptive_num_ctx

        self.assertLess(_adaptive_num_ctx(2_000_000), _adaptive_num_ctx(20_000_000))
        self.assertGreater(_adaptive_num_ctx(20_000_000), 262_144)

    def test_output_reserve_scales_with_input_up_to_the_cap(self) -> None:
        from newme.runner import _adaptive_num_ctx, _OUTPUT_TOKEN_RESERVE_FLOOR

        # A huge input's reserve should exceed the flat floor (scaling kicked
        # in) — detectable via num_ctx landing higher than input-alone would need.
        huge_length = 400_000
        num_ctx = _adaptive_num_ctx(huge_length)
        input_tokens_only_ctx = 1
        while input_tokens_only_ctx < huge_length / 3.5 + _OUTPUT_TOKEN_RESERVE_FLOOR:
            input_tokens_only_ctx *= 2

        self.assertGreaterEqual(num_ctx, input_tokens_only_ctx)

    def test_longer_prompt_never_yields_a_smaller_window(self) -> None:
        from newme.runner import _adaptive_num_ctx

        self.assertLessEqual(_adaptive_num_ctx(1_000), _adaptive_num_ctx(50_000))


if __name__ == "__main__":
    unittest.main()

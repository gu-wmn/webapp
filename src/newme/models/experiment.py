from __future__ import annotations

import random
from datetime import datetime, timezone

from ..extensions import db

VALID_WMN_TYPES = [
    ("WMN: non-understanding", "Non-understanding"),
    ("WMN: disagreement", "Disagreement"),
    ("WMN: other", "Other"),
]

GLOBAL_TEMPLATE_DEFAULT = (
    "You are assisting with linguistic analysis of dialogues to identify and assess "
    "possible Word Meaning Negotiations (WMNs). Work only from the dialogue and any "
    "explicitly provided context. Stay conservative, do not invent evidence, and do "
    "not guess spans, labels, or classifications when the dialogue does not support "
    "them. When spans are requested, keep them as precise as possible. Follow the "
    "requested output format exactly."
)

FREE_TEXT_APPENDIX_DEFAULT = (
    "Respond in plain text. Be concise and ground each observation in specific "
    "utterances from the dialogue. Avoid speculation beyond what the text supports."
)

# Canonical separator the model must use when a hit's quote spans multiple
# utterances — see _QUOTE_FIELD_INSTRUCTIONS below. Kept as a constant because
# portal.py's span-derivation and validation code must parse the exact same token.
MULTI_UTTERANCE_QUOTE_SEPARATOR = " [...] "

_QUOTE_FIELD_INSTRUCTIONS = (
    "the exact verbatim text that motivated the hit — the specific word or "
    "phrase itself, not the entire utterance it occurs in, unless the whole "
    "utterance is genuinely what motivated the hit. If utterance_start_index "
    "equals utterance_end_index, quote only that minimal exact excerpt from "
    "within the utterance. If the hit spans multiple utterances, quote ONLY "
    f'the exact text of the utterance_start_index utterance, followed by "'
    f'{MULTI_UTTERANCE_QUOTE_SEPARATOR.strip()}", followed by the exact text of the '
    "utterance_end_index utterance — do not include text from any utterance "
    "in between, and do not paraphrase, summarize, or alter punctuation or spelling"
)

_LABEL_FIELD_INSTRUCTIONS = (
    'one of "Trigger", "Indicator", or "Negotiation" — only use a label type '
    "that was actually requested earlier in this prompt. Do not emit a label "
    "that wasn't asked for just because it's listed here as a valid value; not "
    "every label type applies to every prompt in the chain"
)

SIMPLIFIED_APPENDIX_DEFAULT = (
    'Output your findings as a JSON object with key "hits" containing an array.\n'
    'Each item must have:\n'
    '  - dialogue_id: the id for the dialogue\n'
    '  - utterance_start_index (integer): inclusive start utterance index in dialogue.utterances\n'
    '  - utterance_end_index (integer): inclusive end utterance index in dialogue.utterances\n'
    f'  - label (string): {_LABEL_FIELD_INSTRUCTIONS}\n'
    f'  - quote (string): {_QUOTE_FIELD_INSTRUCTIONS}\n\n'
    'If no WMN is found, return: {"hits": []}'
)

DETAILED_APPENDIX_DEFAULT = (
    'Output your findings as a JSON object with key "hits" containing an array.\n'
    'Each item must have:\n'
    '  - dialogue_id: the id for the dialogue\n'
    '  - utterance_start_index (integer): inclusive start utterance index in dialogue.utterances\n'
    '  - utterance_end_index (integer): inclusive end utterance index in dialogue.utterances\n'
    f'  - label (string): {_LABEL_FIELD_INSTRUCTIONS}\n'
    f'  - quote (string): {_QUOTE_FIELD_INSTRUCTIONS}\n'
    '  - wmn_type (string): "non-understanding", "disagreement", or "other" — '
    "the classification of the WMN's Indicator. Use the same wmn_type value on "
    "every hit that shares the same wmn_group\n"
    '  - wmn_group (integer): identifies which candidate WMN this hit belongs to '
    "within this dialogue. If the dialogue contains more than one candidate WMN, "
    "give the Trigger, Indicator, and Negotiation hits that together make up the "
    "same WMN the same wmn_group number, and use a different number for each "
    "separate WMN, starting from 1. If only one WMN is found, use 1\n\n"
    'If no WMN is found, return: {"hits": []}'
)

DIALOGUE_INPUT_INSTRUCTIONS_DEFAULT = (
    'Dialogue JSON — an object with a "dialogue" key containing "dialogue_id", '
    '"corpus_codename", and "utterances": an ordered list of '
    "{utterance_index, speaker, text} objects:"
)

REGEX_INPUT_INSTRUCTIONS_DEFAULT = (
    'Regex candidate JSON (optional hints, may be empty) — an object with a "hits" '
    "key containing an array of hit objects with at least utterance_start_index, "
    "utterance_end_index, label, and quote. These come from pattern matching, not "
    "model judgement — treat them as hints only:"
)

PREVIOUS_OUTPUT_INSTRUCTIONS_DEFAULT = (
    'Previous stage JSON (candidate hits from the prior prompt, may be empty) — an '
    'object with a "hits" key, in the same shape as that prompt\'s output format '
    'instructions produce. If a hit has a "wmn_group" field, every hit sharing the '
    "same wmn_group number belongs to the same candidate WMN — treat them as one "
    "linked set, not as independent candidates:"
)

DEFAULT_PROMPT_1_NAME = "Indicator detection"
DEFAULT_PROMPT_1_MODEL = "qwen3:30b"
DEFAULT_PROMPT_1_HOST = "http://merl.clasp.gu.se:11434"
DEFAULT_PROMPT_1_OUTPUT_FORMAT = "detailed"
DEFAULT_PROMPT_1_TEXT = (
    "Your task is to identify candidate Indicators of Word Meaning Negotiation "
    "(WMN). The aim is to detect utterances that signal that the meaning or use "
    "of a specific word or phrase has become an issue in the conversation.\n\n"
    "A WMN involves a meta-linguistic shift: attention moves from the main "
    "topic of the conversation toward the meaning, interpretation, or "
    "appropriateness of a word or phrase. An Indicator is the utterance where "
    "this shift becomes visible — it signals that the meaning or use of a word "
    "or phrase may need to be clarified, questioned, challenged, or "
    "discussed.\n\n"
    "Indicators typically arise in one of two ways, though you don't need to "
    "distinguish between them here:\n"
    "- The speaker appears not to understand a word or phrase and requests or "
    "signals a need for clarification of its meaning.\n"
    "- The speaker appears to challenge the meaning, appropriateness, "
    "interpretation, or applicability of a word or phrase in the current "
    "context.\n\n"
    "For each candidate, determine whether the utterance appears to focus on, "
    "question, request clarification of, or challenge the meaning or use of a "
    "word or phrase. Distinguish semantic or meta-linguistic issues from "
    "simple mishearing, pronunciation, or requests for repetition — cases "
    "that concern only what was heard or how something was pronounced should "
    "normally not be treated as Indicators unless the meaning or use of the "
    "expression is also at issue.\n\n"
    "Be inclusive: a later stage will review each candidate and decide which "
    "are genuine, so prefer including uncertain cases over excluding them. "
    "The utterance doesn't need to be preceded by an identifiable trigger "
    "word, and it doesn't need to be followed by any particular kind of "
    "response — the key question is whether the utterance itself plausibly "
    "functions as a signal that word meaning or word use has become relevant."
)

DEFAULT_PROMPT_2_NAME = "WMN validation"
DEFAULT_PROMPT_2_MODEL = "llama3.3:70b-instruct-q4_K_M"
DEFAULT_PROMPT_2_HOST = "http://merl.clasp.gu.se:11434"
DEFAULT_PROMPT_2_OUTPUT_FORMAT = "detailed"
DEFAULT_PROMPT_2_TEXT = (
    "You are reviewing previously identified candidate Indicator utterances to "
    "determine whether each can be confirmed as part of a genuine Word Meaning "
    "Negotiation (WMN).\n\n"
    "The previous stage deliberately identified Indicators inclusively: a "
    "candidate could be proposed even when its Trigger or subsequent "
    "Negotiation was unclear or absent. In this review stage, apply a "
    "stricter criterion. A candidate should be confirmed as a WMN only when "
    "the surrounding dialogue provides evidence for a complete "
    "Trigger–Indicator–Negotiation structure and a genuine meta-linguistic "
    "shift.\n\n"
    "A WMN occurs when a conversation shifts from its main topic to "
    "discussing the meaning, interpretation, or use of a particular word or "
    "phrase. This meta-linguistic shift distinguishes a WMN from ordinary "
    "discussion of the topic itself.\n\n"
    "A confirmed WMN has three parts:\n\n"
    "Trigger: A preceding utterance containing the specific word or phrase "
    "whose meaning, interpretation, or use is subsequently questioned, "
    "challenged, or requested for clarification. The Trigger may not be "
    "recognisable as such until the Indicator appears.\n\n"
    "Indicator: An utterance signalling that the meaning, interpretation, or "
    "use of a word or phrase has become an issue. The two main forms are:\n"
    "- NON — non-understanding: the speaker does not understand an "
    "expression or requests clarification of what it means.\n"
    "- DIN — disagreement: the speaker challenges the meaning, "
    "interpretation, appropriateness, or applicability of an expression in "
    "the current context.\n"
    "- Other: use only when the utterance clearly functions as a "
    "meta-linguistic Indicator but does not fit NON or DIN.\n\n"
    "Negotiation: One or more subsequent utterances in which participants "
    "address the meaning, interpretation, or use of the Trigger expression. "
    "The Negotiation must provide evidence of a genuine meta-linguistic "
    "shift: attention moves from simply discussing the original topic to "
    "discussing the expression itself. The Negotiation may be intertwined "
    "with continued discussion of the original topic and may span multiple "
    "turns.\n\n"
    "For each candidate Indicator:\n"
    "- Identify the particular word or phrase that the Indicator concerns. "
    "Locate that expression in a preceding utterance. This is the Trigger.\n"
    "- Examine the utterance or utterances following the Indicator. "
    "Determine whether they actually address the meaning, interpretation, "
    "or use of the Trigger expression. These constitute the Negotiation.\n"
    "- Confirm that there is a genuine meta-linguistic shift. It is not "
    "sufficient for the conversation merely to continue discussing the "
    "underlying topic.\n"
    "- Distinguish semantic or meta-linguistic issues from simple "
    "mishearing, requests for repetition, or pronunciation problems. These "
    "are not WMNs unless the meaning, interpretation, or use of the "
    "expression itself also becomes an issue.\n"
    "- Confirm the candidate as a WMN only if a clear Trigger, Indicator, "
    "and Negotiation can all be identified and a genuine meta-linguistic "
    "shift is present. Otherwise, reject it.\n"
    "- For confirmed WMNs, classify the Indicator as NON, DIN, or Other.\n\n"
    "Rejecting a candidate means it should not be included in your results. "
    "This does not mean the original Indicator was unreasonable to propose "
    "— it means the surrounding dialogue does not provide enough evidence "
    "to confirm a complete WMN."
)

DEFAULT_SINGLE_PROMPT_NAME = "Find WMNs"
DEFAULT_SINGLE_PROMPT_MODEL = "qwen3.8:27b-q8_0"
DEFAULT_SINGLE_PROMPT_HOST = "http://lark.clasp.gu.se:11434"
DEFAULT_SINGLE_PROMPT_OUTPUT_FORMAT = "detailed"
DEFAULT_SINGLE_PROMPT_TEXT = """Your task is to identify Word Meaning Negotiations (WMNs) in this dialogue.

A WMN occurs when a conversation shifts from its main topic to discussing the meaning, interpretation, or use of a particular word or phrase. This meta-linguistic shift distinguishes a WMN from ordinary discussion of the topic itself.

A WMN has three parts:

Trigger: A preceding utterance containing the specific word or phrase whose meaning, interpretation, or use is subsequently questioned, challenged, or requested for clarification. The Trigger may not be recognisable as such until the Indicator appears.

Indicator: An utterance signalling that the meaning, interpretation, or use of a word or phrase has become an issue. The two main forms are:

NON — non-understanding: the speaker does not understand an expression or requests clarification of what it means.
DIN — disagreement: the speaker challenges the meaning, interpretation, appropriateness, or applicability of an expression in the current context.
Other: use only when the utterance clearly functions as a meta-linguistic Indicator but does not fit NON or DIN.
Negotiation: One or more subsequent utterances in which participants address the meaning, interpretation, or use of the Trigger expression. The Negotiation must provide evidence of a genuine meta-linguistic shift: attention moves from simply discussing the original topic to discussing the expression itself. The Negotiation may be intertwined with continued discussion of the original topic and may span multiple turns.

For each utterance that could signal a meta-linguistic issue:

Identify the particular word or phrase it concerns, and locate that expression in a preceding utterance. This is the Trigger.
Examine the utterance or utterances that follow. Determine whether they actually address the meaning, interpretation, or use of the Trigger expression. These constitute the Negotiation.
Confirm that there is a genuine meta-linguistic shift. It is not sufficient for the conversation merely to continue discussing the underlying topic.
Distinguish semantic or meta-linguistic issues from simple mishearing, requests for repetition, or pronunciation problems. These are not WMNs unless the meaning, interpretation, or use of the expression itself also becomes an issue.

Confirm it as a WMN only if a clear Trigger, Indicator, and Negotiation can all be identified and a genuine meta-linguistic shift is present. Otherwise, move on.
For each confirmed WMN, classify the Indicator as NON, DIN, or Other.
Only include genuine WMNs in your results — a word or phrase merely being mentioned, or a topic being discussed at length, is not enough on its own."""

REGEX_FORMAT_HELP = (
    'Enter a JSON array of pattern objects. Each object must have:\n'
    '  "label"   — "Indicator", "Trigger", or "Negotiation"\n'
    '  "pattern" — a Python-compatible regular expression\n'
    '  "flags"   — optional string of flags: "i" (ignore case), "s" (dot matches newline)\n'
    'Use \\\\ for a literal backslash in JSON (e.g. "\\\\b" for a word boundary).\n'
    'Regex is best suited for Indicators — explicit, formulaic non-understanding markers.\n'
    'Triggers and Negotiation are context-dependent and better left to the LLM.'
)

REGEX_PATTERNS_DEFAULT = (
    '[\n'
    '  {"label": "Indicator", "pattern": "what(\'s| is) \\\\w+", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "what(\'s| is) that (supposed to )?mean", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "what(\'s| is) the meaning of", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "what(\'s| is) meant by", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "what(\'s| is) (your |a |the )?definition of", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "what do you mean", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "what does (that|this|it|\\\\w+) mean", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "what exactly (do you mean|does that mean)", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "mean by (that|this|it)", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "how do you mean", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "in what sense", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "in what way", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "define \\\\w+", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "how (do you|would you) define", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "how are you using (the |that |this )?\\\\w+", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "what do you call", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "what does that refer to", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "what kind of \\\\w+", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "\\\\byou mean\\\\b", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "do you mean", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "by (that|this|it|\\\\w+) you mean", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "are you saying", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "so you(\'re| are) saying", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "when you say", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "what are you (saying|talking about|getting at|referring to|driving at)", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "if I (understand|follow) (you |correctly|you correctly)", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "could you (clarify|explain|elaborate|rephrase|be more specific|be more precise|spell that out|put it differently)", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "can you (clarify|explain|elaborate|rephrase|be more specific|be more precise|put it differently)", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "please (clarify|explain|elaborate)", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "say (that |it )?again", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "come again", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "I (don\'t|didn\'t|do not|did not) (understand|follow|get (it|that|what you)|see what you mean|know what you mean)", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "I (can\'t|cannot) (follow|understand)", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "I(\'m| am) not (sure what you|following|sure I understand|with you|getting it|getting this)", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "I(\'m| am) not getting (it|this|that|what you)", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "I(\'m| am) (afraid I |sorry I )?(don\'t|do not|didn\'t) understand", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "I(\'m| am) not sure (what|how) (you|that)", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "I(\'m| am) (confused|lost|unclear) (about|on|by|what|how)", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "I don\'t get (it|that|what you)", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "I(\'ve| have) (never|not) heard (of|that)", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "I(\'m| am) not familiar with", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "that(\'s| is) (not clear|unclear|confusing|ambiguous|a bit vague)", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "that doesn\'t make sense", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "(that\'s|it\'s) (a bit |rather |quite )?(unclear|vague|confusing)", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "\\\\?", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "\\\\bpardon\\\\b", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "\\\\bhuh\\\\b", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "what(\'s| is) the difference between", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "\\\\bis not the same as\\\\b", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "are you using .{1,40} to mean", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "in what sense are you using", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "depends (what you mean|how you define|on how you|on what you mean)", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "different definitions? of", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "that(\'s| is) not\\\\b", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "(strange|weird|bizarre|narrow|broad|incorrect|wrong) definition", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "remind (me|us) what (you mean|that means)", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "(let\'s|we (need|have) to|we should) define", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "what exactly (is|was|are|were)", "flags": "i"}\n'
    ']'
)


class UserSettings(db.Model):
    __tablename__ = "user_settings"

    user_email = db.Column(db.String, primary_key=True)
    global_template = db.Column(db.Text, nullable=True)
    free_text_appendix = db.Column(db.Text, nullable=True)
    simplified_appendix = db.Column(db.Text, nullable=True)
    detailed_appendix = db.Column('annotated_appendix', db.Text, nullable=True)
    regex_patterns = db.Column(db.Text, nullable=True)
    dialogue_input_instructions = db.Column(db.Text, nullable=True)
    regex_input_instructions = db.Column(db.Text, nullable=True)
    previous_output_instructions = db.Column(db.Text, nullable=True)

    @property
    def effective_global_template(self) -> str:
        return self.global_template if self.global_template is not None else GLOBAL_TEMPLATE_DEFAULT

    @property
    def effective_free_text_appendix(self) -> str:
        return self.free_text_appendix if self.free_text_appendix is not None else FREE_TEXT_APPENDIX_DEFAULT

    @property
    def effective_simplified_appendix(self) -> str:
        return self.simplified_appendix if self.simplified_appendix is not None else SIMPLIFIED_APPENDIX_DEFAULT

    @property
    def effective_detailed_appendix(self) -> str:
        return self.detailed_appendix if self.detailed_appendix is not None else DETAILED_APPENDIX_DEFAULT

    @property
    def effective_regex_patterns(self) -> str:
        return self.regex_patterns if self.regex_patterns is not None else REGEX_PATTERNS_DEFAULT

    @property
    def effective_dialogue_input_instructions(self) -> str:
        return (
            self.dialogue_input_instructions
            if self.dialogue_input_instructions is not None
            else DIALOGUE_INPUT_INSTRUCTIONS_DEFAULT
        )

    @property
    def effective_regex_input_instructions(self) -> str:
        return (
            self.regex_input_instructions
            if self.regex_input_instructions is not None
            else REGEX_INPUT_INSTRUCTIONS_DEFAULT
        )

    @property
    def effective_previous_output_instructions(self) -> str:
        return (
            self.previous_output_instructions
            if self.previous_output_instructions is not None
            else PREVIOUS_OUTPUT_INSTRUCTIONS_DEFAULT
        )


class Experiment(db.Model):
    __tablename__ = "experiments"

    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String, nullable=False, index=True)
    name = db.Column(db.String, nullable=False)
    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    corpus_filter = db.Column(db.JSON, nullable=False, default=list)
    wmn_type_filter = db.Column(
        db.JSON,
        nullable=False,
        default=lambda: [v for v, _ in VALID_WMN_TYPES],
    )
    sample_size = db.Column(db.Integer, nullable=True, default=5)
    random_seed = db.Column(
        db.Integer,
        nullable=False,
        default=lambda: random.randint(1, 99999),
    )
    dialogues_resolved_at = db.Column(db.DateTime, nullable=True)

    dialogues = db.relationship(
        "ExperimentDialogue",
        backref="experiment",
        cascade="all, delete-orphan",
    )
    prompts = db.relationship(
        "Prompt",
        backref="experiment",
        cascade="all, delete-orphan",
        order_by="Prompt.position",
    )
    runs = db.relationship(
        "Run",
        backref="experiment",
        cascade="all, delete-orphan",
    )
    regex_runs = db.relationship(
        "RegexRun",
        backref="experiment",
        cascade="all, delete-orphan",
    )


class ExperimentDialogue(db.Model):
    __tablename__ = "experiment_dialogues"

    id = db.Column(db.Integer, primary_key=True)
    experiment_id = db.Column(
        db.Integer, db.ForeignKey("experiments.id"), nullable=False, index=True
    )
    dialogue_external_id = db.Column(db.String, nullable=False)
    corpus_codename = db.Column(db.String, nullable=False)


class Prompt(db.Model):
    __tablename__ = "prompts"

    id = db.Column(db.Integer, primary_key=True)
    experiment_id = db.Column(
        db.Integer, db.ForeignKey("experiments.id"), nullable=False, index=True
    )
    position = db.Column(db.Integer, nullable=False)
    name = db.Column(db.String, nullable=False)
    host = db.Column(db.String, nullable=True)
    model = db.Column(db.String, nullable=False)
    include_global_template = db.Column(db.Boolean, nullable=False, default=True)
    include_dialogue = db.Column(db.Boolean, nullable=False, default=True)
    include_regex_candidates = db.Column(db.Boolean, nullable=False, default=False)
    system_prompt = db.Column(db.Text, nullable=True)
    prompt_text = db.Column(db.Text, nullable=False)
    output_format = db.Column(db.String, nullable=True)  # "simplified", "detailed", or None
    temperature = db.Column(db.Float, nullable=True)
    num_ctx = db.Column(db.Integer, nullable=True)

    runs = db.relationship(
        "Run",
        backref="prompt",
        cascade="all, delete-orphan",
    )


class Run(db.Model):
    __tablename__ = "runs"

    id = db.Column(db.Integer, primary_key=True)
    experiment_id = db.Column(
        db.Integer, db.ForeignKey("experiments.id"), nullable=False, index=True
    )
    prompt_id = db.Column(
        db.Integer, db.ForeignKey("prompts.id"), nullable=False, index=True
    )
    status = db.Column(db.String, nullable=False, default="pending")
    total_count = db.Column(db.Integer, nullable=False, default=0)
    processed_count = db.Column(db.Integer, nullable=False, default=0)
    total_char_count = db.Column(db.Integer, nullable=True)
    current_dialogue_char_count = db.Column(db.Integer, nullable=True)
    last_progress_at = db.Column(db.DateTime, nullable=True)
    started_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    last_error = db.Column(db.Text, nullable=True)

    results = db.relationship(
        "RunResult",
        backref="run",
        cascade="all, delete-orphan",
    )


class RunResult(db.Model):
    __tablename__ = "run_results"

    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(
        db.Integer, db.ForeignKey("runs.id"), nullable=False, index=True
    )
    dialogue_external_id = db.Column(db.String, nullable=False)
    corpus_codename = db.Column(db.String, nullable=False)
    output = db.Column(db.JSON, nullable=True)
    raw_response = db.Column(db.Text, nullable=True)
    error = db.Column(db.Text, nullable=True)
    dialogue_char_count = db.Column(db.Integer, nullable=True)
    duration_seconds = db.Column(db.Float, nullable=True)


class RegexRun(db.Model):
    __tablename__ = "regex_runs"

    id = db.Column(db.Integer, primary_key=True)
    experiment_id = db.Column(
        db.Integer, db.ForeignKey("experiments.id"), nullable=False, index=True
    )
    status = db.Column(db.String, nullable=False, default="pending")
    total_count = db.Column(db.Integer, nullable=False, default=0)
    processed_count = db.Column(db.Integer, nullable=False, default=0)
    started_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    error_message = db.Column(db.Text, nullable=True)

    results = db.relationship(
        "RegexRunResult",
        backref="regex_run",
        cascade="all, delete-orphan",
    )


class RegexRunResult(db.Model):
    __tablename__ = "regex_run_results"

    id = db.Column(db.Integer, primary_key=True)
    regex_run_id = db.Column(
        db.Integer, db.ForeignKey("regex_runs.id"), nullable=False, index=True
    )
    dialogue_external_id = db.Column(db.String, nullable=False)
    corpus_codename = db.Column(db.String, nullable=False)
    output = db.Column(db.JSON, nullable=True)
    error = db.Column(db.Text, nullable=True)
    dialogue_char_count = db.Column(db.Integer, nullable=True)
    duration_seconds = db.Column(db.Float, nullable=True)

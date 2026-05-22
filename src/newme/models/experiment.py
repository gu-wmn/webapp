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

SIMPLIFIED_APPENDIX_DEFAULT = (
    'Output your findings as a JSON object with key "hits" containing an array.\n'
    'Each item must have:\n'
    '  - utterance_start_index (integer): inclusive start utterance index in dialogue.utterances\n'
    '  - utterance_end_index (integer): inclusive end utterance index in dialogue.utterances\n'
    '  - label (string): "Trigger", "Indicator", or "Negotiation"\n'
    '  - quote (string): the exact text that motivated the candidate hit\n\n'
    'If no WMN is found, return: {"hits": []}'
)

DETAILED_APPENDIX_DEFAULT = (
    'Output your findings as a JSON object with key "hits" containing an array.\n'
    'Each item must have:\n'
    '  - utterance_start_index (integer): inclusive start utterance index in dialogue.utterances\n'
    '  - utterance_end_index (integer): inclusive end utterance index in dialogue.utterances\n'
    '  - char_start_index (integer): 0-based character offset into the start utterance text\n'
    '  - char_end_index (integer): 0-based exclusive character offset into the end utterance text\n'
    '  - label (string): "Trigger", "Indicator", or "Negotiation"\n'
    '  - quote (string): the exact text covered by the indices\n'
    '  - wmn_type (string): "non-understanding", "disagreement", or "other"\n\n'
    'The quote must exactly match the indexed span.\n'
    'If no WMN is found, return: {"hits": []}'
)

DEFAULT_PROMPT_1_NAME = "Indicator detection"
DEFAULT_PROMPT_1_MODEL = "qwen3:30b"
DEFAULT_PROMPT_1_HOST = "http://merl.clasp.gu.se:11434"
DEFAULT_PROMPT_1_OUTPUT_FORMAT = "simplified"
DEFAULT_PROMPT_1_TEXT = (
    "The inputs below are JSON. The dialogue is in dialogue.utterances. Regex "
    "candidate hits are optional hints only and may be empty.\n\n"
    "Dialogue JSON:\n"
    "{dialogue}\n\n"
    "Regex candidate JSON:\n"
    "{regex_candidates}\n\n"
    "An Indicator is an utterance that signals a need to discuss or clarify the "
    "meaning of a word or phrase. It may take the form of a direct request for "
    "clarification, a challenge to how a word is being used, or an expression of "
    "non-understanding tied to a specific word or phrase.\n\n"
    "Read the dialogue JSON and identify all utterances that could be Indicators. "
    "Include uncertain cases — a later stage will determine which are genuine. "
    "Prefer inclusion over exclusion. Use utterance_start_index and "
    "utterance_end_index to refer to dialogue.utterances. For quote, include the "
    "exact text in the utterance that motivated the candidate hit."
)

DEFAULT_PROMPT_2_NAME = "WMN validation"
DEFAULT_PROMPT_2_MODEL = "llama3.3:70b-instruct-q4_K_M"
DEFAULT_PROMPT_2_HOST = "http://merl.clasp.gu.se:11434"
DEFAULT_PROMPT_2_OUTPUT_FORMAT = "detailed"
DEFAULT_PROMPT_2_TEXT = (
    "The inputs below are JSON. The dialogue is in dialogue.utterances. The previous "
    "stage output is candidate JSON that may be empty. Those candidates are "
    "utterance-level hints, not final character-level spans.\n\n"
    "Dialogue JSON:\n"
    "{dialogue}\n\n"
    "Previous stage JSON:\n"
    "{previous_output}\n\n"
    "You are reviewing candidate Indicator utterances to determine whether each "
    "belongs to a genuine Word Meaning Negotiation (WMN).\n\n"
    "A WMN occurs when a conversation shifts from its main topic to explicitly "
    "discussing the meaning of a word or phrase — a meta-linguistic shift. "
    "This shift is what distinguishes a WMN from ordinary conversation.\n\n"
    "Every WMN follows a three-part structure:\n\n"
    "Trigger: The utterance containing the specific word or phrase whose meaning "
    "later becomes contested. The Trigger precedes the Indicator and may not be "
    "recognisable as such until the Indicator appears.\n\n"
    "Indicator: The utterance signalling that a word's meaning needs to be "
    "discussed. It takes one of two forms:\n"
    "- A clarification request: the listener does not understand the word and asks "
    "for an explanation (NON — non-understanding)\n"
    "- A meta-linguistic objection: the listener challenges the appropriateness or "
    "meaning of the word in the given context (DIN — disagreement)\n\n"
    "Negotiation: One or more response turns following the Indicator where the "
    "meaning is actively discussed or explained. The Negotiation must reflect a "
    "genuine meta-linguistic shift — the focus moves from the original topic to "
    "the word's meaning itself, even if intertwined with the original discussion. "
    "Multiple turns may constitute the Negotiation.\n\n"
    "To confirm a WMN from each candidate Indicator:\n"
    "1. Identify the specific word or phrase being questioned and locate it "
    "precisely in a preceding utterance — this is the Trigger.\n"
    "2. Check whether the response after the Indicator contains a meta-linguistic "
    "shift. If the speaker ignores the question, changes subject, or only "
    "continues the original topic without addressing the word's meaning, it is "
    "not a WMN.\n"
    "3. Confirm the issue is semantic — about what the word means — and not a "
    "mishearing or pronunciation problem.\n"
    "4. If all three parts are present and the meta-linguistic shift is clear, "
    "confirm the WMN and classify it as NON (non-understanding) or DIN "
    "(disagreement), or Other if neither clearly applies.\n\n"
    "For each candidate, work through the steps above. Confirm or reject. "
    "For confirmed WMNs, identify the Trigger word or phrase with its precise "
    "character-level location, the Indicator utterance, and all Negotiation "
    "utterances. Reject any Indicator that does not have a clear Trigger and "
    "Negotiation. Use utterance_start_index and utterance_end_index to refer to "
    "dialogue.utterances, and use character offsets within the utterance text. "
    "char_end_index must be exclusive."
)

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

    regex_enabled = db.Column(db.Boolean, nullable=False, default=False)
    regex_patterns = db.Column(db.Text, nullable=True)

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
    started_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    error_message = db.Column(db.Text, nullable=True)

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

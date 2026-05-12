from __future__ import annotations

import random
from datetime import datetime, timezone

from ..extensions import db

VALID_WMN_TYPES = [
    ("WMN: non-understanding", "Non-understanding"),
    ("WMN: disagreement", "Disagreement"),
    ("WMN: other", "Other"),
]

SIMPLIFIED_APPENDIX_DEFAULT = (
    'Output your findings as a JSON object with key "hits" containing an array.\n'
    'Each item must have:\n'
    '  - dialogue_id (string)\n'
    '  - start_index (integer): utterance index where the span starts\n'
    '  - end_index (integer): utterance index where the span ends\n'
    '  - label (string): "Trigger", "Indicator", or "Negotiation"\n\n'
    'If no WMN is found, return: {"hits": []}'
)

ANNOTATED_APPENDIX_DEFAULT = (
    'Output your findings as a JSON object with key "hits" containing an array.\n'
    'Each item must have:\n'
    '  - dialogue_id (string)\n'
    '  - start_index (integer): utterance index where the span starts\n'
    '  - end_index (integer): utterance index where the span ends\n'
    '  - start_offset (integer): character offset within the start utterance\n'
    '  - end_offset (integer): character offset within the end utterance\n'
    '  - label (string): "Trigger", "Indicator", or "Negotiation"\n'
    '  - excerpt (string): the exact text of the annotated span\n'
    '  - wmn_type (string): "non-understanding", "disagreement", or "other"\n\n'
    'If no WMN is found, return: {"hits": []}'
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
    '  {"label": "Indicator", "pattern": "what do you mean", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "what\'s that mean", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "\\\\byou mean\\\\b", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "in what sense", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "could you (clarify|explain|elaborate)", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "I (don\'t|didn\'t|do not) (understand|follow|see what you mean)", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "what (is|are|do you mean by) (a |an |the )?\\\\w+\\\\?", "flags": "i"},\n'
    '  {"label": "Indicator", "pattern": "define \\\\w+", "flags": "i"}\n'
    ']'
)


class UserSettings(db.Model):
    __tablename__ = "user_settings"

    user_email = db.Column(db.String, primary_key=True)
    global_template = db.Column(db.Text, nullable=True)
    simplified_appendix = db.Column(db.Text, nullable=True)
    annotated_appendix = db.Column(db.Text, nullable=True)
    regex_patterns = db.Column(db.Text, nullable=True)

    @property
    def effective_simplified_appendix(self) -> str:
        return self.simplified_appendix if self.simplified_appendix is not None else SIMPLIFIED_APPENDIX_DEFAULT

    @property
    def effective_annotated_appendix(self) -> str:
        return self.annotated_appendix if self.annotated_appendix is not None else ANNOTATED_APPENDIX_DEFAULT

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
    sample_size = db.Column(db.Integer, nullable=True)
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
    system_prompt = db.Column(db.Text, nullable=True)
    prompt_text = db.Column(db.Text, nullable=False)
    output_format = db.Column(db.String, nullable=True)  # "simplified", "annotated", or None
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

from __future__ import annotations

import functools
import random
import re
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from flask import Blueprint, jsonify, redirect, render_template, request, session, url_for
from sqlalchemy.orm import selectinload

from .extensions import db
from .models.experiment import (
    DEFAULT_PROMPT_1_HOST,
    DEFAULT_PROMPT_1_MODEL,
    DEFAULT_PROMPT_1_NAME,
    DEFAULT_PROMPT_1_OUTPUT_FORMAT,
    DEFAULT_PROMPT_1_TEXT,
    DEFAULT_PROMPT_2_HOST,
    DEFAULT_PROMPT_2_MODEL,
    DEFAULT_PROMPT_2_NAME,
    DEFAULT_PROMPT_2_OUTPUT_FORMAT,
    DEFAULT_PROMPT_2_TEXT,
    DETAILED_APPENDIX_DEFAULT,
    DIALOGUE_INPUT_INSTRUCTIONS_DEFAULT,
    FREE_TEXT_APPENDIX_DEFAULT,
    GLOBAL_TEMPLATE_DEFAULT,
    MULTI_UTTERANCE_QUOTE_SEPARATOR,
    PREVIOUS_OUTPUT_INSTRUCTIONS_DEFAULT,
    REGEX_FORMAT_HELP,
    REGEX_INPUT_INSTRUCTIONS_DEFAULT,
    REGEX_PATTERNS_DEFAULT,
    SIMPLIFIED_APPENDIX_DEFAULT,
    VALID_WMN_TYPES,
    Experiment,
    ExperimentDialogue,
    Prompt,
    RegexRun,
    RegexRunResult,
    Run,
    RunResult,
    UserSettings,
)

bp = Blueprint("portal", __name__)


@bp.app_template_filter("duration")
def format_duration(seconds: float | None) -> str:
    """Render a second count as the two most significant non-zero units
    (d/h/m/s), dropping seconds once hours are shown — mirrors the ETA
    formatting used for in-progress runs.
    """
    if seconds is None:
        return ""
    total = max(0, round(seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


_VALID_WMN_VALUES = {v for v, _ in VALID_WMN_TYPES}
_VALID_LABEL_NAMES = {"Trigger", "Indicator", "Negotiation"}
_RESULT_WMN_MEANINGS = {"both"}


def _get_users() -> dict[str, str]:
    from flask import current_app

    raw = current_app.config.get("USERS", "")
    users: dict[str, str] = {}
    for entry in (raw or "").split(","):
        entry = entry.strip()
        if ":" in entry:
            email, _, password = entry.partition(":")
            users[email.strip().lower()] = password.strip()
    return users


def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("portal.login"))
        return f(*args, **kwargs)

    return decorated


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _hit_int(hit: dict[str, Any], new_key: str, old_key: str) -> int | None:
    value = hit.get(new_key)
    if value is None:
        value = hit.get(old_key)
    return _safe_int(value)


def _hit_text(hit: dict[str, Any], new_key: str, old_key: str) -> str:
    value = hit.get(new_key)
    if value is None:
        value = hit.get(old_key)
    return str(value or "")


def _span_text_from_utterances(
    utterances: list[dict[str, str]],
    start_index: int,
    end_index: int,
    start_offset: int,
    end_offset: int,
) -> str:
    if start_index == end_index:
        return utterances[start_index]["text"][start_offset:end_offset]

    parts = [utterances[start_index]["text"][start_offset:]]
    for idx in range(start_index + 1, end_index):
        parts.append(utterances[idx]["text"])
    parts.append(utterances[end_index]["text"][:end_offset])
    return "\n".join(parts)


def _validate_result_hits(
    result_output: Any,
    utterances: list[dict[str, str]],
    *,
    output_format: str | None,
) -> list[dict[str, str]]:
    if not isinstance(result_output, list) or not utterances:
        return []

    issues: list[dict[str, str]] = []
    for hit_index, hit in enumerate(result_output, start=1):
        if not isinstance(hit, dict):
            continue

        start_index = _hit_int(hit, "utterance_start_index", "start_index")
        end_index = _hit_int(hit, "utterance_end_index", "end_index")
        quote = _hit_text(hit, "quote", "excerpt")

        if start_index is None:
            issues.append({
                "severity": "error",
                "message": f"Hit {hit_index} is missing utterance_start_index.",
            })
            continue
        if end_index is None:
            end_index = start_index
        if start_index < 0 or end_index < 0 or end_index < start_index:
            issues.append({
                "severity": "error",
                "message": f"Hit {hit_index} has an invalid utterance span.",
            })
            continue
        if start_index >= len(utterances) or end_index >= len(utterances):
            issues.append({
                "severity": "error",
                "message": f"Hit {hit_index} points outside the dialogue utterances.",
            })
            continue
        if not quote:
            issues.append({
                "severity": "error",
                "message": f"Hit {hit_index} is missing quote.",
            })
            continue

        if not _derive_spans_from_quote(utterances, start_index, end_index, quote):
            if start_index != end_index:
                message = (
                    f"Hit {hit_index} quote does not match either the full spanned text or "
                    f"the \"<start> {MULTI_UTTERANCE_QUOTE_SEPARATOR.strip()} <end>\" "
                    f"boundary format for a multi-utterance hit: {quote!r}"
                )
            else:
                message = (
                    f"Hit {hit_index} quote does not occur in utterance {start_index}: {quote!r}"
                )
            issues.append({"severity": "warning", "message": message})

    types_by_group: dict[Any, set[str]] = {}
    for hit in result_output:
        if not isinstance(hit, dict):
            continue
        group = hit.get("wmn_group")
        wmn_type = str(hit.get("wmn_type") or "").strip()
        if group is None or not wmn_type:
            continue
        types_by_group.setdefault(group, set()).add(wmn_type)
    for group, types in types_by_group.items():
        if len(types) > 1:
            issues.append({
                "severity": "warning",
                "message": (
                    f"wmn_group {group} hits disagree on wmn_type ({', '.join(sorted(types))}); "
                    "using the Indicator hit's type."
                ),
            })

    return issues



def _annotate_dialogue_utterances(
    utterances: list[dict[str, str]],
    labels: list[dict[str, Any]],
    *,
    anchor_prefix: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    annotated = [{"author": row["author"], "text": row["text"]} for row in utterances]
    label_links: list[dict[str, str]] = []
    inserts: list[tuple[int, int, int]] = []
    label_id = 0

    for label in labels:
        start_index = _hit_int(label, "utterance_start_index", "start_index")
        end_index = _hit_int(label, "utterance_end_index", "end_index")
        start_offset_raw = _hit_int(label, "char_start_index", "start_offset")
        end_offset_raw = _hit_int(label, "char_end_index", "end_offset")
        name = label.get("name")

        if (
            start_index is None
            or end_index is None
            or start_offset_raw is None
            or end_offset_raw is None
            or name not in _VALID_LABEL_NAMES
        ):
            continue
        if start_index < 0 or end_index < 0:
            continue
        if end_index < start_index:
            continue
        if start_index >= len(annotated) or end_index >= len(annotated):
            continue

        anchor_id = f"{anchor_prefix}-{label_id}"
        start_span = f'<span class="{name}" id="{anchor_id}">'
        end_span = "</span>"
        label_links.append({
            "name": str(name),
            "excerpt": _hit_text(label, "quote", "excerpt").strip(),
            "anchor_id": anchor_id,
            "wmn_type": str(label.get("wmn_type") or "").strip(),
            "wmn_group": label.get("wmn_group"),
        })

        add_to_start = 0
        for insert in inserts:
            if insert[0] == start_index and insert[1] < start_offset_raw:
                add_to_start += insert[2]
        start_offset = start_offset_raw + add_to_start

        annotated[start_index]["text"] = (
            annotated[start_index]["text"][:start_offset]
            + start_span
            + annotated[start_index]["text"][start_offset:]
        )
        inserts.append((start_index, start_offset_raw, len(start_span)))
        label_id += 1

        if start_index < end_index:
            annotated[start_index]["text"] += end_span

        if end_index - start_index == 1:
            annotated[start_index + 1]["text"] = start_span + annotated[start_index + 1]["text"]
            inserts.append((start_index + 1, 0, len(start_span)))
        elif end_index - start_index > 1:
            for idx in range(start_index + 1, end_index):
                annotated[idx]["text"] = start_span + annotated[idx]["text"] + end_span
                inserts.append((idx, 0, len(start_span)))
                inserts.append((idx, len(annotated[idx]["text"]), len(end_span)))
            annotated[end_index]["text"] = start_span + annotated[end_index]["text"]
            inserts.append((end_index, 0, len(start_span)))

        add_to_end = 0
        for insert in inserts:
            if insert[0] == end_index and insert[1] < end_offset_raw:
                add_to_end += insert[2]
        end_offset = end_offset_raw + add_to_end

        annotated[end_index]["text"] = (
            annotated[end_index]["text"][:end_offset]
            + end_span
            + annotated[end_index]["text"][end_offset:]
        )
        inserts.append((end_index, end_offset_raw, len(end_span)))

    return annotated, label_links


def _load_dialogue_utterances(corpus_codename: str, dialogue_external_id: str) -> list[dict[str, str]]:
    from .models import Corpus, Dialogue, Utterance

    dialogue = (
        db.session.query(Dialogue)
        .join(Corpus, Dialogue.corpus_id == Corpus.id)
        .filter(Corpus.codename == corpus_codename, Dialogue.external_id == dialogue_external_id)
        .one_or_none()
    )
    if dialogue is None:
        return []

    utterance_rows = (
        Utterance.query.filter_by(dialogue_id=dialogue.id)
        .order_by(Utterance.position.asc())
        .all()
    )
    return [{"author": row.author, "text": row.text} for row in utterance_rows]


def _sequence_label_payload(sequence) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    sorted_rows = sorted(
        sequence.labels,
        key=lambda label: (
            label.start_index if label.start_index is not None else 10_000_000,
            label.start_offset if label.start_offset is not None else 10_000_000,
            label.position,
        ),
    )
    for label in sorted_rows:
        if label.name not in _VALID_LABEL_NAMES:
            continue
        labels.append(
            {
                "name": label.name,
                "utterance_start_index": label.start_index,
                "utterance_end_index": label.end_index,
                "char_start_index": label.start_offset,
                "char_end_index": label.end_offset,
                "quote": label.excerpt,
            }
        )
    return labels


def _derive_spans_from_quote(
    utterances: list[dict[str, str]],
    start_index: int,
    end_index: int,
    quote: str,
) -> list[tuple[int, int, int, int]]:
    """Find all occurrences of quote in utterances[start_index..end_index].

    Returns a list of (utt_start_idx, char_start, utt_end_idx, char_end) tuples.
    char_start is the 0-based offset in utt_start_idx's text.
    char_end is the exclusive offset in utt_end_idx's text.
    """
    if not quote:
        return []

    # Build concatenated text and record where each utterance starts within it.
    utt_starts: list[tuple[int, int]] = []  # (utterance_index, start_pos_in_concat)
    parts: list[str] = []
    pos = 0
    for idx in range(start_index, end_index + 1):
        utt_starts.append((idx, pos))
        parts.append(utterances[idx]["text"])
        pos += len(utterances[idx]["text"]) + 1  # +1 for the "\n" separator

    concat = "\n".join(parts)

    def pos_to_utt(p: int) -> tuple[int, int] | None:
        for utt_idx, utt_pos in utt_starts:
            utt_len = len(utterances[utt_idx]["text"])
            if utt_pos <= p < utt_pos + utt_len:
                return (utt_idx, p - utt_pos)
        return None  # falls on a "\n" separator or out of range

    results: list[tuple[int, int, int, int]] = []
    search_from = 0
    while True:
        found = concat.find(quote, search_from)
        if found == -1:
            break
        # Use found_end - 1 (last char of match) to avoid landing on a separator.
        start_info = pos_to_utt(found)
        end_info = pos_to_utt(found + len(quote) - 1)
        if start_info is not None and end_info is not None:
            utt_s, char_s = start_info
            utt_e, char_e_last = end_info
            results.append((utt_s, char_s, utt_e, char_e_last + 1))
        search_from = found + 1

    if results or start_index == end_index:
        return results

    # Multi-utterance hits are allowed to quote only the boundary utterances —
    # "<start utterance text> MULTI_UTTERANCE_QUOTE_SEPARATOR <end utterance text>" —
    # instead of reproducing every utterance in between verbatim. See
    # MULTI_UTTERANCE_QUOTE_SEPARATOR's docstring in models/experiment.py.
    if MULTI_UTTERANCE_QUOTE_SEPARATOR in quote:
        head, _, tail = quote.partition(MULTI_UTTERANCE_QUOTE_SEPARATOR)
        head, tail = head.strip(), tail.strip()
        start_text = utterances[start_index]["text"]
        end_text = utterances[end_index]["text"]
        head_pos = start_text.find(head) if head else 0
        tail_pos = end_text.find(tail) if tail else 0
        if head_pos != -1 and tail_pos != -1:
            char_start = head_pos
            char_end = (tail_pos + len(tail)) if tail else len(end_text)
            return [(start_index, char_start, end_index, char_end)]

    return []


def _result_label_payload(result: RunResult, utterances: list[dict[str, str]]) -> list[dict[str, Any]]:
    if not isinstance(result.output, list):
        return []

    labels: list[dict[str, Any]] = []
    for hit in result.output:
        if not isinstance(hit, dict):
            continue
        name = hit.get("label")
        if name not in _VALID_LABEL_NAMES:
            continue

        start_index = _hit_int(hit, "utterance_start_index", "start_index")
        end_index = _hit_int(hit, "utterance_end_index", "end_index")
        if start_index is None:
            continue
        if end_index is None:
            end_index = start_index
        if start_index < 0 or end_index < 0 or end_index < start_index:
            continue
        if start_index >= len(utterances) or end_index >= len(utterances):
            continue

        excerpt = _hit_text(hit, "quote", "excerpt")
        wmn_type = str(hit.get("wmn_type") or "").strip()
        wmn_group = hit.get("wmn_group")

        occurrences = _derive_spans_from_quote(utterances, start_index, end_index, excerpt)
        if occurrences:
            for utt_s, char_s, utt_e, char_e in occurrences:
                labels.append({
                    "name": name,
                    "utterance_start_index": utt_s,
                    "utterance_end_index": utt_e,
                    "char_start_index": char_s,
                    "char_end_index": char_e,
                    "quote": excerpt,
                    "wmn_type": wmn_type,
                    "wmn_group": wmn_group,
                })
        else:
            # Quote not locatable; fall back to highlighting the full utterance range.
            labels.append({
                "name": name,
                "utterance_start_index": start_index,
                "utterance_end_index": end_index,
                "char_start_index": 0,
                "char_end_index": len(utterances[end_index]["text"]),
                "quote": excerpt,
                "wmn_type": wmn_type,
                "wmn_group": wmn_group,
            })

    return labels


def _canonical_wmn_type(items: list[dict[str, Any]]) -> str:
    """The Indicator's wmn_type if present, else any item's — a WMN has one
    type, but individual hits can disagree with each other (see
    _group_label_links_by_wmn), so the Indicator's is taken as authoritative
    since that's the one the model was actually asked to classify.
    """
    return next(
        (item["wmn_type"] for item in items if item.get("name") == "Indicator" and item.get("wmn_type")),
        next((item["wmn_type"] for item in items if item.get("wmn_type")), ""),
    )


def _group_label_links_by_wmn(label_links: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group chip-row entries by wmn_group, preserving first-seen order.

    Returns a list of {"group": <wmn_group value, or None>, "wmn_type": <canonical
    type or "">, "links": [...]}. When no entry carries a wmn_group (older runs,
    or the simplified output format, which doesn't have the field), everything
    lands in one None-group bucket, and the template renders it exactly like the
    old flat chip row.

    A WMN has one type, but the model reports wmn_type per hit and can disagree
    with itself across a group's Trigger/Indicator/Negotiation hits. wmn_type is
    conceptually a property of the Indicator (the prompt asks the model to
    "classify the Indicator"), so that hit's value is taken as the group's
    canonical type and stamped onto every link in the group — the UI must never
    show two different types for the same WMN.
    """
    if not any(link.get("wmn_group") is not None for link in label_links):
        return [{"group": None, "wmn_type": "", "links": label_links}]

    groups: dict[Any, list[dict[str, Any]]] = {}
    order: list[Any] = []
    ungrouped: list[dict[str, Any]] = []
    for link in label_links:
        group = link.get("wmn_group")
        if group is None:
            ungrouped.append(link)
            continue
        if group not in groups:
            groups[group] = []
            order.append(group)
        groups[group].append(link)

    grouped = []
    for group in order:
        links = groups[group]
        canonical_type = _canonical_wmn_type(links)
        for link in links:
            link["wmn_type"] = canonical_type
        grouped.append({"group": group, "wmn_type": canonical_type, "links": links})
    if ungrouped:
        grouped.append({"group": None, "wmn_type": "", "links": ungrouped})
    return grouped


@bp.get("/login")
@bp.post("/login")
def login():
    if session.get("user"):
        return redirect(url_for("portal.experiments_home"))

    error = None
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        users = _get_users()
        if email in users and users[email] == password:
            session.permanent = True
            session["user"] = email
            return redirect(url_for("portal.experiments_home"))
        error = "Invalid email or password."

    return render_template("portal/login.html", error=error)


@bp.post("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("portal.login"))


@bp.get("/experiments")
@login_required
def experiments_home():
    user = session["user"]
    experiments = (
        Experiment.query.filter_by(user_email=user)
        .order_by(Experiment.created_at.desc())
        .all()
    )
    return render_template(
        "portal/experiments.html",
        user=user,
        experiments=experiments,
    )


def _one_line_preview(text: str | None, limit: int = 100) -> str:
    preview = " ".join((text or "").split())
    if len(preview) > limit:
        preview = preview[:limit].rstrip() + "…"
    return preview


def _settings_field_meta(value: str | None, default: str) -> dict[str, Any]:
    is_custom = bool(value)
    effective = value or default
    return {"is_custom": is_custom, "preview": _one_line_preview(effective)}


def _appendix_options(user_settings: "UserSettings | None") -> dict[str, dict[str, Any]]:
    fields = {
        "": (
            user_settings.free_text_appendix if user_settings else None,
            FREE_TEXT_APPENDIX_DEFAULT,
        ),
        "simplified": (
            user_settings.simplified_appendix if user_settings else None,
            SIMPLIFIED_APPENDIX_DEFAULT,
        ),
        "detailed": (
            user_settings.detailed_appendix if user_settings else None,
            DETAILED_APPENDIX_DEFAULT,
        ),
    }
    options = {}
    for key, (value, default) in fields.items():
        meta = _settings_field_meta(value, default)
        options[key] = {"text": value or default, **meta}
    return options


def _prompt_input_preview(
    experiment: "Experiment", position: int, user_settings: "UserSettings | None"
) -> dict[str, Any]:
    from .models import Corpus, Dialogue, Utterance
    from .runner import _get_previous_output, _get_regex_candidates, format_dialogue

    dialogue_header = (
        user_settings.effective_dialogue_input_instructions
        if user_settings
        else DIALOGUE_INPUT_INSTRUCTIONS_DEFAULT
    )
    regex_header = (
        user_settings.effective_regex_input_instructions
        if user_settings
        else REGEX_INPUT_INSTRUCTIONS_DEFAULT
    )
    previous_header = (
        user_settings.effective_previous_output_instructions
        if user_settings
        else PREVIOUS_OUTPUT_INSTRUCTIONS_DEFAULT
    )

    empty_hits_json = '{"hits": []}'
    regex_empty_text = regex_header.strip() + "\n" + empty_hits_json
    previous_empty_text = previous_header.strip() + "\n" + empty_hits_json

    preview: dict[str, Any] = {
        "sample_dialogue": experiment.dialogues[0] if experiment.dialogues else None,
        "dialogue_text": None,
        "dialogue_preview": "",
        "regex_text": regex_empty_text,
        "regex_preview": _one_line_preview(regex_empty_text),
        "previous_text": previous_empty_text,
        "previous_preview": _one_line_preview(previous_empty_text),
        "previous_prompt": None,
    }

    if position > 1:
        preview["previous_prompt"] = Prompt.query.filter_by(
            experiment_id=experiment.id, position=position - 1
        ).first()

    sample = preview["sample_dialogue"]
    if sample is None:
        return preview

    dialogue_obj = (
        db.session.query(Dialogue)
        .join(Corpus, Dialogue.corpus_id == Corpus.id)
        .filter(
            Corpus.codename == sample.corpus_codename,
            Dialogue.external_id == sample.dialogue_external_id,
        )
        .first()
    )
    if dialogue_obj is None:
        return preview

    utterances = (
        Utterance.query.filter_by(dialogue_id=dialogue_obj.id)
        .order_by(Utterance.position.asc())
        .all()
    )

    preview["dialogue_text"] = dialogue_header.strip() + "\n" + format_dialogue(
        utterances, sample.dialogue_external_id, sample.corpus_codename
    )
    preview["dialogue_preview"] = _one_line_preview(preview["dialogue_text"])

    preview["regex_text"] = regex_header.strip() + "\n" + _get_regex_candidates(
        experiment, sample.dialogue_external_id, utterances
    )
    preview["regex_preview"] = _one_line_preview(preview["regex_text"])

    if position > 1:
        fake_prompt = SimpleNamespace(position=position, experiment_id=experiment.id)
        preview["previous_text"] = previous_header.strip() + "\n" + _get_previous_output(
            fake_prompt, sample.dialogue_external_id, utterances
        )
        preview["previous_preview"] = _one_line_preview(preview["previous_text"])

    return preview


@bp.get("/settings")
@login_required
def settings():
    user = session["user"]
    user_settings = db.session.get(UserSettings, user)
    field_meta = {
        "global_template": _settings_field_meta(
            user_settings.global_template if user_settings else None, GLOBAL_TEMPLATE_DEFAULT
        ),
        "regex_patterns": _settings_field_meta(
            user_settings.regex_patterns if user_settings else None, REGEX_PATTERNS_DEFAULT
        ),
        "dialogue_input_instructions": _settings_field_meta(
            user_settings.dialogue_input_instructions if user_settings else None,
            DIALOGUE_INPUT_INSTRUCTIONS_DEFAULT,
        ),
        "regex_input_instructions": _settings_field_meta(
            user_settings.regex_input_instructions if user_settings else None,
            REGEX_INPUT_INSTRUCTIONS_DEFAULT,
        ),
        "previous_output_instructions": _settings_field_meta(
            user_settings.previous_output_instructions if user_settings else None,
            PREVIOUS_OUTPUT_INSTRUCTIONS_DEFAULT,
        ),
        "free_text_appendix": _settings_field_meta(
            user_settings.free_text_appendix if user_settings else None, FREE_TEXT_APPENDIX_DEFAULT
        ),
        "simplified_appendix": _settings_field_meta(
            user_settings.simplified_appendix if user_settings else None, SIMPLIFIED_APPENDIX_DEFAULT
        ),
        "detailed_appendix": _settings_field_meta(
            user_settings.detailed_appendix if user_settings else None, DETAILED_APPENDIX_DEFAULT
        ),
    }
    return render_template(
        "portal/settings.html",
        user=user,
        us=user_settings,
        field_meta=field_meta,
        global_template_default=GLOBAL_TEMPLATE_DEFAULT,
        free_text_appendix_default=FREE_TEXT_APPENDIX_DEFAULT,
        simplified_appendix_default=SIMPLIFIED_APPENDIX_DEFAULT,
        detailed_appendix_default=DETAILED_APPENDIX_DEFAULT,
        regex_patterns_default=REGEX_PATTERNS_DEFAULT,
        dialogue_input_instructions_default=DIALOGUE_INPUT_INSTRUCTIONS_DEFAULT,
        regex_input_instructions_default=REGEX_INPUT_INSTRUCTIONS_DEFAULT,
        previous_output_instructions_default=PREVIOUS_OUTPUT_INSTRUCTIONS_DEFAULT,
        regex_format_help=REGEX_FORMAT_HELP,
    )


@bp.post("/settings")
@login_required
def save_settings():
    user = session["user"]
    user_settings = db.session.get(UserSettings, user)
    if user_settings is None:
        user_settings = UserSettings(user_email=user)
        db.session.add(user_settings)

    user_settings.global_template = (request.form.get("global_template") or "").strip() or None
    user_settings.free_text_appendix = (request.form.get("free_text_appendix") or "").strip() or None
    user_settings.simplified_appendix = (request.form.get("simplified_appendix") or "").strip() or None
    user_settings.detailed_appendix = (request.form.get("detailed_appendix") or "").strip() or None
    user_settings.regex_patterns = (request.form.get("regex_patterns") or "").strip() or None
    user_settings.dialogue_input_instructions = (
        request.form.get("dialogue_input_instructions") or ""
    ).strip() or None
    user_settings.regex_input_instructions = (
        request.form.get("regex_input_instructions") or ""
    ).strip() or None
    user_settings.previous_output_instructions = (
        request.form.get("previous_output_instructions") or ""
    ).strip() or None
    db.session.commit()
    return redirect(url_for("portal.settings"))


_RESETTABLE_SETTINGS_FIELDS = {
    "global_template",
    "regex_patterns",
    "dialogue_input_instructions",
    "regex_input_instructions",
    "previous_output_instructions",
    "free_text_appendix",
    "simplified_appendix",
    "detailed_appendix",
}


@bp.post("/settings/reset")
@login_required
def reset_settings_field():
    user = session["user"]
    field = (request.form.get("field") or "").strip()
    if field not in _RESETTABLE_SETTINGS_FIELDS:
        return jsonify({"error": "Unknown field"}), 400

    user_settings = db.session.get(UserSettings, user)
    if user_settings is not None:
        setattr(user_settings, field, None)
        db.session.commit()

    return jsonify({"ok": True})


@bp.get("/experiments/new")
@bp.post("/experiments/new")
@login_required
def new_experiment():
    user = session["user"]
    error = None
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        if not name:
            error = "Name is required."
        else:
            exp = Experiment(user_email=user, name=name)
            db.session.add(exp)
            db.session.flush()
            db.session.add(Prompt(
                experiment_id=exp.id,
                position=1,
                name=DEFAULT_PROMPT_1_NAME,
                host=DEFAULT_PROMPT_1_HOST,
                model=DEFAULT_PROMPT_1_MODEL,
                include_global_template=True,
                prompt_text=DEFAULT_PROMPT_1_TEXT,
                output_format=DEFAULT_PROMPT_1_OUTPUT_FORMAT,
            ))
            db.session.add(Prompt(
                experiment_id=exp.id,
                position=2,
                name=DEFAULT_PROMPT_2_NAME,
                host=DEFAULT_PROMPT_2_HOST,
                model=DEFAULT_PROMPT_2_MODEL,
                include_global_template=True,
                prompt_text=DEFAULT_PROMPT_2_TEXT,
                output_format=DEFAULT_PROMPT_2_OUTPUT_FORMAT,
            ))
            db.session.commit()
            _resolve_dialogues(exp)
            return redirect(url_for("portal.experiment", experiment_id=exp.id))
    return render_template("portal/new_experiment.html", user=user, error=error)


@bp.get("/experiments/<int:experiment_id>")
@login_required
def experiment(experiment_id: int):
    user = session["user"]
    exp = Experiment.query.filter_by(id=experiment_id, user_email=user).first_or_404()
    corpora = _get_corpora()
    user_settings = db.session.get(UserSettings, user)

    latest_runs = {
        p.id: Run.query.filter_by(prompt_id=p.id)
        .order_by(Run.started_at.desc())
        .first()
        for p in exp.prompts
    }
    active_run = Run.query.filter(
        Run.experiment_id == exp.id,
        Run.status.in_(["pending", "running"]),
    ).first()
    latest_regex_run = (
        RegexRun.query.filter_by(experiment_id=exp.id)
        .order_by(RegexRun.started_at.desc())
        .first()
    )
    active_regex_run = RegexRun.query.filter(
        RegexRun.experiment_id == exp.id,
        RegexRun.status.in_(["pending", "running"]),
    ).first()

    regex_meta = _settings_field_meta(
        user_settings.regex_patterns if user_settings else None, REGEX_PATTERNS_DEFAULT
    )

    return render_template(
        "portal/experiment.html",
        user=user,
        experiment=exp,
        user_settings=user_settings,
        regex_meta=regex_meta,
        regex_patterns_text=user_settings.effective_regex_patterns if user_settings else REGEX_PATTERNS_DEFAULT,
        corpora=corpora,
        wmn_type_options=VALID_WMN_TYPES,
        latest_runs=latest_runs,
        active_run=active_run,
        latest_regex_run=latest_regex_run,
        active_regex_run=active_regex_run,
    )


@bp.post("/experiments/<int:experiment_id>/delete")
@login_required
def delete_experiment(experiment_id: int):
    user = session["user"]
    exp = Experiment.query.filter_by(id=experiment_id, user_email=user).first_or_404()
    db.session.delete(exp)
    db.session.commit()
    return redirect(url_for("portal.experiments_home"))


@bp.post("/experiments/<int:experiment_id>/configure")
@login_required
def configure_experiment(experiment_id: int):
    user = session["user"]
    exp = Experiment.query.filter_by(id=experiment_id, user_email=user).first_or_404()

    corpus_filter = request.form.getlist("corpus")
    wmn_type_filter = [v for v in request.form.getlist("wmn_type") if v in _VALID_WMN_VALUES]
    if not wmn_type_filter:
        wmn_type_filter = list(_VALID_WMN_VALUES)

    sample_size_raw = (request.form.get("sample_size") or "").strip()
    try:
        sample_size = int(sample_size_raw) if sample_size_raw else None
        if sample_size is not None and sample_size <= 0:
            sample_size = None
    except ValueError:
        sample_size = exp.sample_size

    seed_raw = (request.form.get("random_seed") or "").strip()
    try:
        random_seed = int(seed_raw)
    except ValueError:
        random_seed = exp.random_seed

    exp.corpus_filter = corpus_filter
    exp.wmn_type_filter = wmn_type_filter
    exp.sample_size = sample_size
    exp.random_seed = random_seed
    exp.dialogues_resolved_at = None
    db.session.commit()
    _resolve_dialogues(exp)

    return redirect(url_for("portal.experiment", experiment_id=exp.id))


@bp.post("/experiments/<int:experiment_id>/resolve")
@login_required
def resolve_experiment(experiment_id: int):
    user = session["user"]
    exp = Experiment.query.filter_by(id=experiment_id, user_email=user).first_or_404()
    _resolve_dialogues(exp)
    return redirect(url_for("portal.experiment", experiment_id=exp.id))


@bp.get("/experiments/<int:experiment_id>/dialogues")
@login_required
def experiment_dialogues(experiment_id: int):
    user = session["user"]
    exp = Experiment.query.filter_by(id=experiment_id, user_email=user).first_or_404()

    dialogues = (
        ExperimentDialogue.query.filter_by(experiment_id=exp.id)
        .order_by(ExperimentDialogue.dialogue_external_id)
        .all()
    )

    return render_template(
        "portal/experiment_dialogues.html",
        user=user,
        experiment=exp,
        dialogues=dialogues,
    )


@bp.get("/experiments/<int:experiment_id>/dialogues/<int:dialogue_id>")
@login_required
def experiment_dialogue_detail(experiment_id: int, dialogue_id: int):
    from .models import AnnotationSequence

    user = session["user"]
    exp = Experiment.query.filter_by(id=experiment_id, user_email=user).first_or_404()
    dialogue = ExperimentDialogue.query.filter_by(id=dialogue_id, experiment_id=exp.id).first_or_404()

    utterances = _load_dialogue_utterances(dialogue.corpus_codename, dialogue.dialogue_external_id)

    human_sequences = (
        AnnotationSequence.query.options(selectinload(AnnotationSequence.labels))
        .filter_by(
            corpus_codename=dialogue.corpus_codename,
            dialogue_external_id=dialogue.dialogue_external_id,
        )
        .order_by(AnnotationSequence.wmn_id.asc())
        .all()
    )
    human_sequences = [
        sequence
        for sequence in human_sequences
        if sequence.wmn_type in _VALID_WMN_VALUES
        or (sequence.wmn_meaning or "").strip().lower() in _RESULT_WMN_MEANINGS
    ]

    selected_wmn_id = (request.args.get("wmn_id") or "").strip()
    selected_sequence = None
    if selected_wmn_id:
        selected_sequence = next(
            (sequence for sequence in human_sequences if sequence.wmn_id == selected_wmn_id),
            None,
        )
    if selected_sequence is None and human_sequences:
        selected_sequence = human_sequences[0]

    human_utterances: list[dict[str, str]] = []
    human_label_links: list[dict[str, str]] = []
    if selected_sequence is not None and utterances:
        human_labels = _sequence_label_payload(selected_sequence)
        human_utterances, human_label_links = _annotate_dialogue_utterances(
            utterances,
            human_labels,
            anchor_prefix="human-label",
        )

    return render_template(
        "portal/experiment_dialogue_detail.html",
        user=user,
        experiment=exp,
        dialogue=dialogue,
        utterances=utterances,
        human_sequences=human_sequences,
        selected_sequence=selected_sequence,
        human_utterances=human_utterances,
        human_label_links=human_label_links,
    )


def _resolve_dialogues(exp: Experiment) -> None:
    from .models import AnnotationSequence

    wmn_types = exp.wmn_type_filter or list(_VALID_WMN_VALUES)

    query = db.session.query(
        AnnotationSequence.dialogue_external_id,
        AnnotationSequence.corpus_codename,
    ).filter(
        AnnotationSequence.wmn_type.in_(wmn_types)
    ).distinct()

    if exp.corpus_filter:
        query = query.filter(AnnotationSequence.corpus_codename.in_(exp.corpus_filter))

    pool = sorted((row.dialogue_external_id, row.corpus_codename) for row in query.all())

    rng = random.Random(exp.random_seed)
    if exp.sample_size and exp.sample_size < len(pool):
        selected = rng.sample(pool, exp.sample_size)
    else:
        selected = pool

    ExperimentDialogue.query.filter_by(experiment_id=exp.id).delete()
    for dialogue_external_id, corpus_codename in selected:
        db.session.add(ExperimentDialogue(
            experiment_id=exp.id,
            dialogue_external_id=dialogue_external_id,
            corpus_codename=corpus_codename,
        ))

    exp.dialogues_resolved_at = datetime.now(timezone.utc)
    db.session.commit()


# Regex is disabled for now — nothing calls this, kept so it can be wired back
# up (e.g. behind the same run-section UI) without rebuilding the logic.
def _trigger_regex_run(exp: Experiment) -> None:
    from flask import current_app

    from .runner import execute_regex_run

    if not exp.dialogues:
        return

    active = RegexRun.query.filter(
        RegexRun.experiment_id == exp.id,
        RegexRun.status.in_(["pending", "running"]),
    ).first()
    if active:
        return

    active_llm = Run.query.filter(
        Run.experiment_id == exp.id,
        Run.status.in_(["pending", "running"]),
    ).first()
    if active_llm:
        return

    regex_run = RegexRun(experiment_id=exp.id, total_count=len(exp.dialogues))
    db.session.add(regex_run)
    db.session.commit()

    execute_regex_run(regex_run.id, current_app._get_current_object())


@bp.get("/api/hosts")
@login_required
def api_hosts():
    from flask import current_app, jsonify

    hosts = _get_ollama_hosts(current_app)
    return jsonify({"hosts": hosts})


@bp.get("/api/models")
@login_required
def api_models():
    from flask import current_app, jsonify

    from .ollama_client import get_client

    hosts = _get_ollama_hosts(current_app)
    valid_urls = {h["url"] for h in hosts}

    host_url = request.args.get("host", "").strip()
    if not host_url:
        host_url = hosts[0]["url"] if hosts else ""
    if host_url not in valid_urls:
        return jsonify({"error": "Unknown host", "url": host_url}), 400

    try:
        client = get_client(host_url)
        response = client.list()
        models = sorted(model.model for model in response.models)
        return jsonify({"models": models, "url": host_url})
    except Exception as exc:
        return jsonify({"models": [], "url": host_url, "error": str(exc)}), 502


@bp.get("/experiments/<int:experiment_id>/prompts/new")
@bp.post("/experiments/<int:experiment_id>/prompts/new")
@login_required
def new_prompt(experiment_id: int):
    user = session["user"]
    exp = Experiment.query.filter_by(id=experiment_id, user_email=user).first_or_404()
    user_settings = db.session.get(UserSettings, user) or UserSettings()

    next_position = (
        db.session.query(db.func.max(Prompt.position)).filter_by(experiment_id=exp.id).scalar() or 0
    ) + 1

    error = None
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        host = (request.form.get("host") or "").strip()
        model = (request.form.get("model") or "").strip()
        prompt_text = (request.form.get("prompt_text") or "").strip()

        output_format_raw = request.form.get("output_format") or None
        if not name or not prompt_text or not host or not model:
            error = "Name, prompt text, host, and model are required."
        else:
            temp_raw = (request.form.get("temperature") or "").strip()
            try:
                temperature = float(temp_raw) if temp_raw else None
            except ValueError:
                temperature = None

            ctx_raw = (request.form.get("num_ctx") or "").strip()
            try:
                num_ctx = int(ctx_raw) if ctx_raw else None
            except ValueError:
                num_ctx = None

            output_format = output_format_raw if output_format_raw == "detailed" else None

            db.session.add(Prompt(
                experiment_id=exp.id,
                position=next_position,
                name=name,
                host=host or None,
                model=model or "—",
                include_global_template=request.form.get("include_global_template") == "on",
                include_dialogue=next_position <= 1 or request.form.get("include_dialogue") == "on",
                include_regex_candidates=request.form.get("include_regex_candidates") == "on",
                output_format=output_format,
                prompt_text=prompt_text,
                temperature=temperature,
                num_ctx=num_ctx,
            ))
            db.session.commit()
            return redirect(url_for("portal.experiment", experiment_id=exp.id))

    return render_template(
        "portal/new_prompt.html",
        user=user,
        experiment=exp,
        prompt=None,
        user_settings=user_settings,
        position=next_position,
        global_meta=_settings_field_meta(
            user_settings.global_template if user_settings else None, GLOBAL_TEMPLATE_DEFAULT
        ),
        appendix_options=_appendix_options(user_settings),
        input_preview=_prompt_input_preview(exp, next_position, user_settings),
        error=error,
    )


@bp.get("/experiments/<int:experiment_id>/prompts/<int:prompt_id>/edit")
@bp.post("/experiments/<int:experiment_id>/prompts/<int:prompt_id>/edit")
@login_required
def edit_prompt(experiment_id: int, prompt_id: int):
    user = session["user"]
    exp = Experiment.query.filter_by(id=experiment_id, user_email=user).first_or_404()
    prompt = Prompt.query.filter_by(id=prompt_id, experiment_id=exp.id).first_or_404()
    user_settings = db.session.get(UserSettings, user) or UserSettings()

    error = None
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        host = (request.form.get("host") or "").strip()
        model = (request.form.get("model") or "").strip()
        prompt_text = (request.form.get("prompt_text") or "").strip()

        output_format_raw = request.form.get("output_format") or None
        if not name or not prompt_text or not host or not model:
            error = "Name, prompt text, host, and model are required."
        else:
            prompt.name = name
            prompt.host = host or None
            prompt.model = model or "—"
            prompt.prompt_text = prompt_text
            prompt.include_global_template = request.form.get("include_global_template") == "on"
            prompt.include_dialogue = (
                prompt.position <= 1 or request.form.get("include_dialogue") == "on"
            )
            prompt.include_regex_candidates = request.form.get("include_regex_candidates") == "on"
            prompt.system_prompt = None

            output_format = output_format_raw if output_format_raw == "detailed" else None
            prompt.output_format = output_format

            temp_raw = (request.form.get("temperature") or "").strip()
            try:
                prompt.temperature = float(temp_raw) if temp_raw else None
            except ValueError:
                pass

            ctx_raw = (request.form.get("num_ctx") or "").strip()
            try:
                prompt.num_ctx = int(ctx_raw) if ctx_raw else None
            except ValueError:
                pass

            db.session.commit()
            return redirect(url_for("portal.experiment", experiment_id=exp.id))

    return render_template(
        "portal/edit_prompt.html",
        user=user,
        experiment=exp,
        prompt=prompt,
        user_settings=user_settings,
        position=prompt.position,
        global_meta=_settings_field_meta(
            user_settings.global_template if user_settings else None, GLOBAL_TEMPLATE_DEFAULT
        ),
        appendix_options=_appendix_options(user_settings),
        input_preview=_prompt_input_preview(exp, prompt.position, user_settings),
        error=error,
    )


@bp.post("/experiments/<int:experiment_id>/regex/run")
@login_required
def start_regex_run(experiment_id: int):
    from flask import current_app

    from .runner import execute_regex_run

    user = session["user"]
    exp = Experiment.query.filter_by(id=experiment_id, user_email=user).first_or_404()

    if not exp.dialogues_resolved_at:
        return jsonify({"error": "Resolve the dialogue sample before running."}), 400

    active = RegexRun.query.filter(
        RegexRun.experiment_id == exp.id,
        RegexRun.status.in_(["pending", "running"]),
    ).first()
    if active:
        return jsonify({"error": "A regex run is already in progress.", "run_id": active.id}), 409

    active_llm = Run.query.filter(
        Run.experiment_id == exp.id,
        Run.status.in_(["pending", "running"]),
    ).first()
    if active_llm:
        return jsonify({"error": "An LLM run is already in progress."}), 409

    regex_run = RegexRun(experiment_id=exp.id, total_count=len(exp.dialogues))
    db.session.add(regex_run)
    db.session.commit()

    execute_regex_run(regex_run.id, current_app._get_current_object())
    return jsonify({"run_id": regex_run.id, "total_count": regex_run.total_count})


@bp.get("/api/regex-runs/<int:run_id>/status")
@login_required
def regex_run_status(run_id: int):
    regex_run = db.session.get(RegexRun, run_id)
    if regex_run is None:
        return jsonify({"error": "Not found"}), 404
    Experiment.query.filter_by(
        id=regex_run.experiment_id, user_email=session["user"]
    ).first_or_404()

    return jsonify({
        "run_id": regex_run.id,
        "status": regex_run.status,
        "processed_count": regex_run.processed_count,
        "total_count": regex_run.total_count,
        "error_message": regex_run.error_message,
    })


@bp.post("/api/regex-runs/<int:run_id>/abort")
@login_required
def abort_regex_run(run_id: int):
    regex_run = db.session.get(RegexRun, run_id)
    if regex_run is None:
        return jsonify({"error": "Not found"}), 404
    Experiment.query.filter_by(
        id=regex_run.experiment_id, user_email=session["user"]
    ).first_or_404()

    if regex_run.status not in ("pending", "running"):
        return jsonify({"error": "Run is not active.", "status": regex_run.status}), 409

    regex_run.status = "aborted"
    regex_run.error_message = "Aborted by user."
    regex_run.completed_at = datetime.now(timezone.utc)
    db.session.commit()

    return jsonify({"run_id": regex_run.id, "status": regex_run.status})


@bp.get("/experiments/<int:experiment_id>/regex/results")
@login_required
def regex_run_results(experiment_id: int):
    user = session["user"]
    exp = Experiment.query.filter_by(id=experiment_id, user_email=user).first_or_404()

    regex_run = (
        RegexRun.query.filter_by(experiment_id=exp.id, status="complete")
        .order_by(RegexRun.completed_at.desc())
        .first_or_404()
    )
    results = (
        RegexRunResult.query.filter_by(regex_run_id=regex_run.id)
        .order_by(RegexRunResult.dialogue_external_id)
        .all()
    )
    hit_count = sum(len(r.output) if isinstance(r.output, list) else 0 for r in results)
    metrics, per_result_metrics = _compute_run_metrics(results)

    return render_template(
        "portal/regex_results.html",
        user=user,
        experiment=exp,
        regex_run=regex_run,
        results=results,
        hit_count=hit_count,
        metrics=metrics,
        per_result_metrics=per_result_metrics,
    )


@bp.get("/experiments/<int:experiment_id>/regex/results/<int:result_id>")
@login_required
def regex_run_result_dialogue(experiment_id: int, result_id: int):
    from .models import AnnotationSequence

    user = session["user"]
    exp = Experiment.query.filter_by(id=experiment_id, user_email=user).first_or_404()

    regex_run = (
        RegexRun.query.filter_by(experiment_id=exp.id, status="complete")
        .order_by(RegexRun.completed_at.desc())
        .first_or_404()
    )
    result = RegexRunResult.query.filter_by(id=result_id, regex_run_id=regex_run.id).first_or_404()

    utterances = _load_dialogue_utterances(result.corpus_codename, result.dialogue_external_id)
    llm_labels = _result_label_payload(result, utterances)
    llm_validation_issues = _validate_result_hits(
        result.output,
        utterances,
        output_format="detailed",
    )
    llm_utterances: list[dict[str, str]] = []
    llm_label_links: list[dict[str, str]] = []
    if utterances:
        llm_utterances, llm_label_links = _annotate_dialogue_utterances(
            utterances, llm_labels, anchor_prefix="regex-label"
        )

    human_sequences = (
        AnnotationSequence.query.options(selectinload(AnnotationSequence.labels))
        .filter_by(
            corpus_codename=result.corpus_codename,
            dialogue_external_id=result.dialogue_external_id,
        )
        .order_by(AnnotationSequence.wmn_id.asc())
        .all()
    )
    human_sequences = [
        s for s in human_sequences
        if s.wmn_type in _VALID_WMN_VALUES
        or (s.wmn_meaning or "").strip().lower() in _RESULT_WMN_MEANINGS
    ]

    selected_wmn_id = (request.args.get("wmn_id") or "").strip()
    selected_sequence = None
    if selected_wmn_id:
        selected_sequence = next(
            (s for s in human_sequences if s.wmn_id == selected_wmn_id), None
        )
    if selected_sequence is None and human_sequences:
        selected_sequence = human_sequences[0]

    human_utterances: list[dict[str, str]] = []
    human_labels: list[dict[str, Any]] = []
    human_label_links: list[dict[str, str]] = []
    if selected_sequence is not None and utterances:
        human_labels = _sequence_label_payload(selected_sequence)
        human_utterances, human_label_links = _annotate_dialogue_utterances(
            utterances, human_labels, anchor_prefix="human-label"
        )

    return render_template(
        "portal/regex_result_dialogue.html",
        user=user,
        experiment=exp,
        regex_run=regex_run,
        result=result,
        utterances_missing=not utterances,
        llm_hit_count=len(llm_labels),
        llm_validation_issues=llm_validation_issues,
        llm_utterances=llm_utterances,
        llm_labels=llm_labels,
        llm_label_links=llm_label_links,
        human_sequences=human_sequences,
        selected_sequence=selected_sequence,
        human_utterances=human_utterances,
        human_labels=human_labels,
        human_label_links=human_label_links,
    )


@bp.post("/experiments/<int:experiment_id>/prompts/<int:prompt_id>/delete")
@login_required
def delete_prompt(experiment_id: int, prompt_id: int):
    user = session["user"]
    exp = Experiment.query.filter_by(id=experiment_id, user_email=user).first_or_404()
    prompt = Prompt.query.filter_by(id=prompt_id, experiment_id=exp.id).first_or_404()
    db.session.delete(prompt)
    db.session.commit()
    _renumber_prompts(exp.id)
    return redirect(url_for("portal.experiment", experiment_id=exp.id))


def _renumber_prompts(experiment_id: int) -> None:
    prompts = (
        Prompt.query.filter_by(experiment_id=experiment_id)
        .order_by(Prompt.position)
        .all()
    )
    for i, prompt in enumerate(prompts, start=1):
        prompt.position = i
    db.session.commit()


@bp.post("/experiments/<int:experiment_id>/prompts/<int:prompt_id>/run")
@login_required
def start_run(experiment_id: int, prompt_id: int):
    from flask import current_app

    from .runner import compute_experiment_char_count, execute_run

    user = session["user"]
    exp = Experiment.query.filter_by(id=experiment_id, user_email=user).first_or_404()
    prompt = Prompt.query.filter_by(id=prompt_id, experiment_id=exp.id).first_or_404()

    if not exp.dialogues_resolved_at:
        return jsonify({"error": "Resolve the dialogue sample before running."}), 400

    active = Run.query.filter(
        Run.experiment_id == exp.id,
        Run.status.in_(["pending", "running"]),
    ).first()
    if active:
        return jsonify({"error": "A run is already in progress.", "run_id": active.id}), 409

    run = Run(
        experiment_id=exp.id,
        prompt_id=prompt.id,
        total_count=len(exp.dialogues),
        total_char_count=compute_experiment_char_count(exp),
    )
    db.session.add(run)
    db.session.commit()

    execute_run(run.id, current_app._get_current_object())
    return jsonify({"run_id": run.id, "total_count": run.total_count})


@bp.get("/api/runs/<int:run_id>/status")
@login_required
def run_status(run_id: int):
    run = db.session.get(Run, run_id)
    if run is None:
        return jsonify({"error": "Not found"}), 404
    Experiment.query.filter_by(
        id=run.experiment_id, user_email=session["user"]
    ).first_or_404()

    run_eta_at = None
    run_eta_source = None
    dialogue_eta_at = None
    dialogue_eta_source = None
    if run.status in ("pending", "running"):
        from .runner import estimate_current_dialogue_eta, estimate_run_eta

        prompt = db.session.get(Prompt, run.prompt_id)
        run_eta_dt, run_eta_source = estimate_run_eta(run, prompt)
        run_eta_at = run_eta_dt.isoformat() if run_eta_dt else None
        dialogue_eta_dt, dialogue_eta_source = estimate_current_dialogue_eta(run, prompt)
        dialogue_eta_at = dialogue_eta_dt.isoformat() if dialogue_eta_dt else None

    return jsonify({
        "run_id": run.id,
        "status": run.status,
        "processed_count": run.processed_count,
        "total_count": run.total_count,
        "error_message": run.error_message,
        "last_error": run.last_error if run.status in ("pending", "running") else None,
        "run_eta_at": run_eta_at,
        "run_eta_source": run_eta_source,
        "dialogue_eta_at": dialogue_eta_at,
        "dialogue_eta_source": dialogue_eta_source,
    })


@bp.post("/api/runs/<int:run_id>/abort")
@login_required
def abort_run(run_id: int):
    run = db.session.get(Run, run_id)
    if run is None:
        return jsonify({"error": "Not found"}), 404
    Experiment.query.filter_by(
        id=run.experiment_id, user_email=session["user"]
    ).first_or_404()

    if run.status not in ("pending", "running"):
        return jsonify({"error": "Run is not active.", "status": run.status}), 409

    run.status = "aborted"
    run.error_message = "Aborted by user."
    run.last_error = None
    run.completed_at = datetime.now(timezone.utc)
    db.session.commit()

    from .runner import request_ollama_abort
    request_ollama_abort(run.id)

    return jsonify({"run_id": run.id, "status": run.status})


def _human_label_instances_with_sequence(sequences) -> list[tuple[str, int, int, str]]:
    """Return (label_name, start_index, end_index, sequence_wmn_id) for each valid human label."""
    instances: list[tuple[str, int, int, str]] = []
    for seq in sequences:
        for label in seq.labels:
            if label.name not in _VALID_LABEL_NAMES:
                continue
            if label.start_index is None or label.end_index is None:
                continue
            instances.append((label.name, label.start_index, label.end_index, seq.wmn_id))
    return instances


def _human_label_instances(sequences) -> list[tuple[str, int, int]]:
    """Return (label_name, start_index, end_index) for each valid human annotation label."""
    return [
        (name, start, end)
        for name, start, end, _seq in _human_label_instances_with_sequence(sequences)
    ]


def _llm_hit_instances_with_group(result: RunResult) -> list[tuple[str | None, int, int, int | None]]:
    """Return (label_name, start_index, end_index, wmn_group) for each LLM output hit.

    label_name may be None or an unrecognized value — such hits can never match a
    human instance (see _ranges_overlap callers) but still count toward false positives.
    """
    if not isinstance(result.output, list):
        return []
    instances: list[tuple[str | None, int, int, int | None]] = []
    for hit in result.output:
        if not isinstance(hit, dict):
            continue
        start = hit.get("utterance_start_index", hit.get("start_index"))
        end = hit.get("utterance_end_index", hit.get("end_index"))
        if start is None:
            continue
        end = end if end is not None else start
        try:
            start, end = int(start), int(end)
        except (TypeError, ValueError):
            continue
        if end < start:
            continue
        instances.append((hit.get("label"), start, end, hit.get("wmn_group")))
    return instances


def _llm_hit_instances(result: RunResult) -> list[tuple[str | None, int, int]]:
    """Return (label_name, start_index, end_index) for each LLM output hit."""
    return [
        (name, start, end)
        for name, start, end, _group in _llm_hit_instances_with_group(result)
    ]


def _ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and b_start <= a_end


def _match_wmn_groups_to_sequences(
    human_instances: list[tuple[str, int, int, str]],
    llm_instances: list[tuple[str | None, int, int, int | None]],
) -> dict[str, Any]:
    """Pair each model wmn_group with at most one human WMN sequence, and vice versa.

    Only the Indicator span decides whether a group and a sequence are "the same"
    WMN — the Indicator is the anchor of a WMN, so two candidates match if their
    Indicators overlap in any way, regardless of whether their Trigger/Negotiation
    spans line up too. Candidate pairs are scored by how many overlapping
    Indicator pairs they share (normally 0 or 1, since a well-formed group/sequence
    each carry a single Indicator) and assigned highest-scoring-first (greedy
    max-weight bipartite matching), skipping a pair once either side has already
    been claimed. With typically only a handful of WMNs per dialogue this greedy
    approach reaches the optimal assignment in practice without needing a real
    matching solver.

    Returns:
        {
            "group_to_sequence": {wmn_group: sequence_wmn_id, ...},
            "sequence_to_group": {sequence_wmn_id: wmn_group, ...},
            "unmatched_groups": [wmn_group, ...] with no pairing, sorted,
            "unmatched_sequences": [sequence_wmn_id, ...] with no pairing, sorted,
        }
    """
    groups = sorted({group for _, _, _, group in llm_instances if group is not None})
    sequences = sorted({seq for _, _, _, seq in human_instances if seq is not None})

    scores: dict[tuple[Any, Any], int] = {}
    for llm_name, llm_start, llm_end, group in llm_instances:
        if group is None or llm_name != "Indicator":
            continue
        for h_name, h_start, h_end, seq in human_instances:
            if seq is None or h_name != "Indicator":
                continue
            if _ranges_overlap(llm_start, llm_end, h_start, h_end):
                key = (group, seq)
                scores[key] = scores.get(key, 0) + 1

    ordered_pairs = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))

    group_to_sequence: dict[Any, Any] = {}
    sequence_to_group: dict[Any, Any] = {}
    for (group, seq), _score in ordered_pairs:
        if group in group_to_sequence or seq in sequence_to_group:
            continue
        group_to_sequence[group] = seq
        sequence_to_group[seq] = group

    return {
        "group_to_sequence": group_to_sequence,
        "sequence_to_group": sequence_to_group,
        "unmatched_groups": [g for g in groups if g not in group_to_sequence],
        "unmatched_sequences": [s for s in sequences if s not in sequence_to_group],
    }


def _match_hits_to_human_instances(
    human_instances: list[tuple[str, int, int]],
    llm_instances: list[tuple[str | None, int, int]],
) -> dict[str, int]:
    """Classify hits/human instances for one dialogue via label-aware overlap matching.

    A hit is a true positive if some human instance shares its label and its
    utterance range overlaps the hit's — no matter where the quote falls, and no
    matter how many other utterances the hit's span covers. A human instance is
    "recalled" under the same rule, checked independently of the hit side (see
    _compute_run_metrics docstring for why the two directions aren't required to
    produce the same count).
    """
    result_hit_tp = 0
    human_matched = [False] * len(human_instances)
    for llm_name, llm_start, llm_end in llm_instances:
        matched_this_hit = False
        for i, (h_name, h_start, h_end) in enumerate(human_instances):
            if h_name != llm_name:
                continue
            if _ranges_overlap(llm_start, llm_end, h_start, h_end):
                matched_this_hit = True
                human_matched[i] = True
        if matched_this_hit:
            result_hit_tp += 1

    result_human_tp = sum(1 for m in human_matched if m)

    return {
        "hit_tp": result_hit_tp,
        "hit_fp": len(llm_instances) - result_hit_tp,
        "human_tp": result_human_tp,
        "human_fn": len(human_instances) - result_human_tp,
    }


_WORD_RE = re.compile(r"[A-Za-z']+")


def _quote_word_overlap(a: str, b: str) -> bool:
    """True if two quotes share at least one word, case-insensitive.

    Used for Trigger matching, which cares whether the same contested word was
    identified — not whether the two quotes were anchored to the same utterance,
    since a model and a human annotator can reasonably point at different
    occurrences of the same word.
    """
    words_a = {w.lower() for w in _WORD_RE.findall(a or "")}
    words_b = {w.lower() for w in _WORD_RE.findall(b or "")}
    return bool(words_a & words_b)


def _llm_hit_records(result: RunResult) -> list[dict[str, Any]]:
    """Return {name, start, end, group, quote} for each LLM output hit."""
    if not isinstance(result.output, list):
        return []
    records: list[dict[str, Any]] = []
    for hit in result.output:
        if not isinstance(hit, dict):
            continue
        start = hit.get("utterance_start_index", hit.get("start_index"))
        end = hit.get("utterance_end_index", hit.get("end_index"))
        if start is None:
            continue
        end = end if end is not None else start
        try:
            start, end = int(start), int(end)
        except (TypeError, ValueError):
            continue
        if end < start:
            continue
        records.append({
            "name": hit.get("label"),
            "start": start,
            "end": end,
            "group": hit.get("wmn_group"),
            "quote": _hit_text(hit, "quote", "excerpt"),
        })
    return records


def _human_label_records(sequences) -> list[dict[str, Any]]:
    """Return {name, start, end, sequence, quote} for each valid human label."""
    records: list[dict[str, Any]] = []
    for seq in sequences:
        for label in seq.labels:
            if label.name not in _VALID_LABEL_NAMES:
                continue
            if label.start_index is None or label.end_index is None:
                continue
            records.append({
                "name": label.name,
                "start": label.start_index,
                "end": label.end_index,
                "sequence": seq.wmn_id,
                "quote": label.excerpt or "",
            })
    return records


def _labels_match(
    label_name: str, llm_items: list[dict[str, Any]], human_items: list[dict[str, Any]]
) -> bool:
    """Does any llm item of this label match any human item of the same label?

    Trigger matches on shared quote word, position-independent. Indicator and
    Negotiation match on utterance-range overlap — for Negotiation specifically,
    any part of any negotiation-labeled utterance overlapping is enough.
    """
    for llm_item in llm_items:
        for human_item in human_items:
            if label_name == "Trigger":
                if _quote_word_overlap(llm_item["quote"], human_item["quote"]):
                    return True
            elif _ranges_overlap(
                llm_item["start"], llm_item["end"], human_item["start"], human_item["end"]
            ):
                return True
    return False


def _match_wmn_label_instances(
    human_records: list[dict[str, Any]],
    llm_records: list[dict[str, Any]],
) -> dict[str, int]:
    """Classify hits/human instances for one dialogue, counting once per (WMN, label).

    Unlike _match_hits_to_human_instances, which treats every individual hit and
    every individual human label row as its own unit, this counts each WMN's
    Trigger, Indicator, and Negotiation as a single slot to match or miss — a
    human WMN annotated with three separate Trigger rows (a real pattern in this
    data: the same contested word marked at several points in the dialogue)
    still counts as one Trigger to recall, not three.

    WMN pairing reuses _match_wmn_groups_to_sequences (Indicator-overlap only,
    strict 1:1). Within a paired (group, sequence), each label type present on
    either side is then checked with _labels_match. A group with no pairing
    contributes a false positive for every label type it has; an unpaired
    sequence contributes a false negative for every label type it has — an
    unmatched WMN is wrong (or missed) across the board, not just on its
    Indicator.
    """
    llm_tuples = [(r["name"], r["start"], r["end"], r["group"]) for r in llm_records]
    human_tuples = [(r["name"], r["start"], r["end"], r["sequence"]) for r in human_records]
    wmn_match = _match_wmn_groups_to_sequences(human_tuples, llm_tuples)

    def _by_group(group: Any) -> list[dict[str, Any]]:
        return [r for r in llm_records if r["group"] == group]

    def _by_sequence(seq: Any) -> list[dict[str, Any]]:
        return [r for r in human_records if r["sequence"] == seq]

    def _label_names(items: list[dict[str, Any]]) -> list[str]:
        return sorted({r["name"] for r in items if r["name"] in _VALID_LABEL_NAMES})

    groups = sorted({r["group"] for r in llm_records if r["group"] is not None})
    sequences = sorted({r["sequence"] for r in human_records if r["sequence"] is not None})

    hit_tp = hit_fp = 0
    for group in groups:
        group_items = _by_group(group)
        seq = wmn_match["group_to_sequence"].get(group)
        seq_items = _by_sequence(seq) if seq is not None else []
        for label_name in _label_names(group_items):
            llm_items = [r for r in group_items if r["name"] == label_name]
            human_items = [r for r in seq_items if r["name"] == label_name]
            if human_items and _labels_match(label_name, llm_items, human_items):
                hit_tp += 1
            else:
                hit_fp += 1

    human_tp = human_fn = 0
    for seq in sequences:
        seq_items = _by_sequence(seq)
        group = wmn_match["sequence_to_group"].get(seq)
        group_items = _by_group(group) if group is not None else []
        for label_name in _label_names(seq_items):
            human_items = [r for r in seq_items if r["name"] == label_name]
            llm_items = [r for r in group_items if r["name"] == label_name]
            if llm_items and _labels_match(label_name, llm_items, human_items):
                human_tp += 1
            else:
                human_fn += 1

    return {
        "hit_tp": hit_tp,
        "hit_fp": hit_fp,
        "human_tp": human_tp,
        "human_fn": human_fn,
    }


def _compute_run_metrics(results: list[RunResult]) -> tuple[dict, dict[int, dict]]:
    """Compute precision/recall/F1 against human annotations, per dialogue.

    For dialogues where the model output carries a wmn_group (the "detailed"
    output format), label-level matching happens per-WMN via
    _match_wmn_label_instances: each WMN's Trigger, Indicator, and Negotiation
    is one slot to match or miss, not one per raw hit/label row — see that
    function's docstring for the per-label matching rules and why a
    group/sequence with no WMN pairing counts as wrong (or missed) on every
    label it has. WMN-level matching (whole WMNs, not individual labels) reuses
    the same _match_wmn_groups_to_sequences pairing (Indicator-overlap only,
    strict 1:1): a human WMN is "matched" if some model group paired with it at
    all, regardless of how many of its labels that group actually got right.

    For dialogues without a wmn_group anywhere (older runs, or the retired
    "simplified" format, which doesn't have the field), label-level matching
    falls back to _match_hits_to_human_instances, and WMN-level matching is
    unavailable (there's no group to pair on).

    Precision and recall are independently classified — precision counts model
    hits/groups as true/false positives, recall counts human instances/sequences
    as recalled/missed — so their numerators aren't required to be equal.

    Returns:
        aggregate: dict with precision, recall, f1, tp, fp, recall_tp, fn,
                   wmn_matched, wmn_human_total, dialogues_with_human
        per_result: dict mapping result.id ->
                     {wmn_matched, wmn_human_total, wmn_model_extra}
                     (wmn_matched/wmn_human_total are None when the dialogue's
                     output has no wmn_group to pair on)
    """
    from .models import AnnotationSequence

    # Batch-load all relevant AnnotationSequences
    keys = {(r.corpus_codename, r.dialogue_external_id) for r in results}
    all_seqs = (
        AnnotationSequence.query
        .options(selectinload(AnnotationSequence.labels))
        .filter(
            db.tuple_(
                AnnotationSequence.corpus_codename,
                AnnotationSequence.dialogue_external_id,
            ).in_(list(keys))
        )
        .all()
    )
    # Filter to relevant sequences (same logic as run_result_dialogue)
    seqs_by_dialogue: dict[tuple, list] = {}
    for seq in all_seqs:
        if seq.wmn_type not in _VALID_WMN_VALUES and (seq.wmn_meaning or "").strip().lower() not in _RESULT_WMN_MEANINGS:
            continue
        key = (seq.corpus_codename, seq.dialogue_external_id)
        seqs_by_dialogue.setdefault(key, []).append(seq)

    hit_tp = hit_fp = 0
    human_tp = human_fn = 0
    wmn_matched_total = wmn_human_total_total = 0
    dialogues_with_hits = dialogues_with_human = dialogues_both = 0
    per_result: dict[int, dict] = {}

    for result in results:
        key = (result.corpus_codename, result.dialogue_external_id)
        human_seqs = seqs_by_dialogue.get(key, [])
        llm_records = _llm_hit_records(result)

        wmn_matched = wmn_human_total = None
        wmn_model_extra = 0

        if any(r["group"] is not None for r in llm_records):
            human_records = _human_label_records(human_seqs)
            match = _match_wmn_label_instances(human_records, llm_records)
            has_hits = bool(llm_records)
            has_human = bool(human_records)

            llm_tuples = [(r["name"], r["start"], r["end"], r["group"]) for r in llm_records]
            human_tuples = [(r["name"], r["start"], r["end"], r["sequence"]) for r in human_records]
            wmn_match = _match_wmn_groups_to_sequences(human_tuples, llm_tuples)
            wmn_matched = len(wmn_match["group_to_sequence"])
            wmn_human_total = len(human_seqs)
            wmn_model_extra = len(wmn_match["unmatched_groups"])
            wmn_matched_total += wmn_matched
            wmn_human_total_total += wmn_human_total
        else:
            human_instances = _human_label_instances(human_seqs)
            llm_instances = _llm_hit_instances(result)
            match = _match_hits_to_human_instances(human_instances, llm_instances)
            has_hits = bool(llm_instances)
            has_human = bool(human_instances)

        hit_tp += match["hit_tp"]
        hit_fp += match["hit_fp"]
        human_tp += match["human_tp"]
        human_fn += match["human_fn"]

        if has_hits:
            dialogues_with_hits += 1
        if has_human:
            dialogues_with_human += 1
        if has_hits and has_human:
            dialogues_both += 1

        per_result[result.id] = {
            "wmn_matched": wmn_matched,
            "wmn_human_total": wmn_human_total,
            "wmn_model_extra": wmn_model_extra,
        }

    precision = hit_tp / (hit_tp + hit_fp) if (hit_tp + hit_fp) > 0 else None
    recall = human_tp / (human_tp + human_fn) if (human_tp + human_fn) > 0 else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None

    aggregate = {
        "tp": hit_tp, "fp": hit_fp,
        "recall_tp": human_tp, "fn": human_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "wmn_matched": wmn_matched_total,
        "wmn_human_total": wmn_human_total_total,
        "dialogues_with_hits": dialogues_with_hits,
        "dialogues_with_human": dialogues_with_human,
        "dialogues_both": dialogues_both,
        "total": len(results),
    }
    return aggregate, per_result


@bp.get("/experiments/<int:experiment_id>/prompts/<int:prompt_id>/results")
@login_required
def run_results(experiment_id: int, prompt_id: int):
    user = session["user"]
    exp = Experiment.query.filter_by(id=experiment_id, user_email=user).first_or_404()
    prompt = Prompt.query.filter_by(id=prompt_id, experiment_id=exp.id).first_or_404()

    run = (
        Run.query.filter_by(prompt_id=prompt.id)
        .filter(Run.status.in_(("complete", "aborted")))
        .order_by(Run.completed_at.desc())
        .first_or_404()
    )
    results = (
        RunResult.query.filter_by(run_id=run.id)
        .order_by(RunResult.dialogue_external_id)
        .all()
    )
    hit_count = sum(
        len(r.output) if isinstance(r.output, list) else 0
        for r in results
    )
    metrics, per_result_metrics = _compute_run_metrics(results)
    return render_template(
        "portal/results.html",
        user=user,
        experiment=exp,
        prompt=prompt,
        run=run,
        results=results,
        hit_count=hit_count,
        metrics=metrics,
        per_result_metrics=per_result_metrics,
    )


@bp.get("/experiments/<int:experiment_id>/prompts/<int:prompt_id>/results/<int:result_id>")
@login_required
def run_result_dialogue(experiment_id: int, prompt_id: int, result_id: int):
    from .models import AnnotationSequence

    user = session["user"]
    exp = Experiment.query.filter_by(id=experiment_id, user_email=user).first_or_404()
    prompt = Prompt.query.filter_by(id=prompt_id, experiment_id=exp.id).first_or_404()

    result = (
        RunResult.query.join(Run, RunResult.run_id == Run.id)
        .filter(RunResult.id == result_id, Run.prompt_id == prompt.id)
        .first_or_404()
    )
    run = db.session.get(Run, result.run_id)

    utterances = _load_dialogue_utterances(result.corpus_codename, result.dialogue_external_id)
    llm_labels = _result_label_payload(result, utterances)
    llm_validation_issues = _validate_result_hits(
        result.output,
        utterances,
        output_format=prompt.output_format,
    )
    llm_hit_count = len(llm_labels)

    human_sequences = (
        AnnotationSequence.query.options(selectinload(AnnotationSequence.labels))
        .filter_by(
            corpus_codename=result.corpus_codename,
            dialogue_external_id=result.dialogue_external_id,
        )
        .order_by(AnnotationSequence.wmn_id.asc())
        .all()
    )
    human_sequences = [
        sequence
        for sequence in human_sequences
        if sequence.wmn_type in _VALID_WMN_VALUES
        or (sequence.wmn_meaning or "").strip().lower() in _RESULT_WMN_MEANINGS
    ]

    wmn_match = _match_wmn_groups_to_sequences(
        _human_label_instances_with_sequence(human_sequences),
        _llm_hit_instances_with_group(result),
    )
    group_to_sequence = wmn_match["group_to_sequence"]
    sequence_to_group = wmn_match["sequence_to_group"]

    # Unified tabs, ordered matched pairs first, then human-only, then
    # model-only — within each category, model groups keep first-seen order
    # and human sequences keep their existing (wmn_id) order. This puts a
    # single tab over both panels for every matched pair, plus a tab each for
    # whichever side has no counterpart.
    group_order: list[Any] = []
    seen_groups: set[Any] = set()
    for label in llm_labels:
        group = label.get("wmn_group")
        if group is not None and group not in seen_groups:
            seen_groups.add(group)
            group_order.append(group)

    sequence_by_id = {sequence.wmn_id: sequence for sequence in human_sequences}

    matched_tabs: list[dict[str, Any]] = []
    model_only_tabs: list[dict[str, Any]] = []
    for group in group_order:
        entry = {"group": group, "wmn_id": group_to_sequence.get(group)}
        (matched_tabs if entry["wmn_id"] is not None else model_only_tabs).append(entry)

    human_only_tabs = [
        {"group": None, "wmn_id": sequence.wmn_id}
        for sequence in human_sequences
        if sequence.wmn_id not in sequence_to_group
    ]

    tabs: list[dict[str, Any]] = matched_tabs + human_only_tabs + model_only_tabs

    for index, tab in enumerate(tabs, start=1):
        tab["label"] = f"WMN-{index}"
        if tab["group"] is not None and tab["wmn_id"] is not None:
            tab["kind"] = "matched"
        elif tab["group"] is not None:
            tab["kind"] = "model-only"
        else:
            tab["kind"] = "human-only"

    req_group_raw = (request.args.get("group") or "").strip()
    try:
        req_group: Any = int(req_group_raw) if req_group_raw else None
    except ValueError:
        req_group = None
    req_wmn_id = (request.args.get("wmn_id") or "").strip() or None

    selected_tab = None
    if req_group is not None or req_wmn_id is not None:
        selected_tab = next(
            (t for t in tabs if t["group"] == req_group and t["wmn_id"] == req_wmn_id),
            None,
        )
    if selected_tab is None and tabs:
        selected_tab = tabs[0]

    selected_group = selected_tab["group"] if selected_tab else None
    selected_sequence = sequence_by_id.get(selected_tab["wmn_id"]) if selected_tab else None

    # The anchor tab (from the tab bar) never moves once picked — its own
    # side always keeps showing its own dialogue. Only the empty side (the
    # one that fell back to "Compare with" cards) is customizable: picking a
    # card there adds a compare_group/compare_wmn_id param that swaps in
    # that WMN's dialogue for JUST that side, without disturbing the anchor.
    req_compare_group_raw = (request.args.get("compare_group") or "").strip()
    try:
        req_compare_group: Any = int(req_compare_group_raw) if req_compare_group_raw else None
    except ValueError:
        req_compare_group = None
    req_compare_wmn_id = (request.args.get("compare_wmn_id") or "").strip() or None

    effective_group = selected_group
    effective_sequence = selected_sequence
    comparing_llm = False
    comparing_human = False

    if selected_tab and selected_tab["kind"] == "human-only" and req_compare_group is not None:
        compare_tab = next(
            (t for t in tabs if t["kind"] == "model-only" and t["group"] == req_compare_group),
            None,
        )
        if compare_tab is not None:
            effective_group = compare_tab["group"]
            comparing_llm = True

    if selected_tab and selected_tab["kind"] == "model-only" and req_compare_wmn_id is not None:
        compare_sequence = sequence_by_id.get(req_compare_wmn_id)
        if compare_sequence is not None and compare_sequence.wmn_id not in sequence_to_group:
            effective_sequence = compare_sequence
            comparing_human = True

    # Older results (pre-wmn_group, e.g. the retired "simplified" output format)
    # carry no group info to pair on — fall back to showing every hit
    # regardless of the selected tab, matching how they rendered before tabs.
    if group_order:
        group_llm_labels = [
            label for label in llm_labels
            if effective_group is not None and label.get("wmn_group") == effective_group
        ]
    else:
        group_llm_labels = llm_labels
    # Only show a side's dialogue text when that side actually has something
    # selected — a human-only tab has no model group to highlight, and a
    # model-only tab has no human sequence, so there's nothing meaningful to
    # show there, unless a "Compare with" card filled it in. The legacy
    # (pre-wmn_group) fallback is the exception: with no group info to pair
    # on at all, every hit is shown regardless of tab.
    llm_utterances: list[dict[str, str]] = []
    llm_label_links: list[dict[str, str]] = []
    if utterances and (not group_order or effective_group is not None):
        llm_utterances, llm_label_links = _annotate_dialogue_utterances(
            utterances,
            group_llm_labels,
            anchor_prefix="llm-label",
        )
    llm_label_groups = _group_label_links_by_wmn(llm_label_links) if llm_label_links else []

    human_utterances: list[dict[str, str]] = []
    human_label_links: list[dict[str, str]] = []
    if utterances and effective_sequence is not None:
        human_labels = _sequence_label_payload(effective_sequence)
        human_utterances, human_label_links = _annotate_dialogue_utterances(
            utterances,
            human_labels,
            anchor_prefix="human-label",
        )

    # "Compare with" cards: when a tab has nothing on one side, that side
    # shows cards for the other side's non-matching WMNs instead of an empty
    # dialogue column — a quick way to see what the model/human found
    # elsewhere in this dialogue that didn't pair with anything.
    model_only_cards = [
        {
            "tab": tab,
            "wmn_type": _canonical_wmn_type(
                [label for label in llm_labels if label.get("wmn_group") == tab["group"]]
            ),
            "labels": [label for label in llm_labels if label.get("wmn_group") == tab["group"]],
        }
        for tab in tabs
        if tab["kind"] == "model-only"
    ]
    human_only_cards = [
        {
            "tab": tab,
            "wmn_type": sequence_by_id[tab["wmn_id"]].wmn_type,
            "labels": _sequence_label_payload(sequence_by_id[tab["wmn_id"]]),
        }
        for tab in tabs
        if tab["kind"] == "human-only"
    ]

    return render_template(
        "portal/result_dialogue.html",
        user=user,
        experiment=exp,
        prompt=prompt,
        run=run,
        result=result,
        utterances_missing=not utterances,
        llm_hit_count=llm_hit_count,
        llm_validation_issues=llm_validation_issues,
        llm_utterances=llm_utterances,
        llm_labels=llm_labels,
        llm_label_links=llm_label_links,
        llm_label_groups=llm_label_groups,
        human_sequences=human_sequences,
        tabs=tabs,
        selected_tab=selected_tab,
        selected_sequence=effective_sequence,
        comparing_llm=comparing_llm,
        comparing_human=comparing_human,
        human_utterances=human_utterances,
        human_label_links=human_label_links,
        model_only_cards=model_only_cards,
        human_only_cards=human_only_cards,
    )


def _get_ollama_hosts(app) -> list[dict]:
    raw = app.config.get("OLLAMA_HOSTS", "")
    fallback = app.config.get("OLLAMA_URL", "http://127.0.0.1:11434")
    hosts = []
    for entry in (raw or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "|" in entry:
            label, _, url = entry.partition("|")
            hosts.append({"label": label.strip(), "url": url.strip()})
        else:
            hosts.append({"label": entry, "url": entry})
    if not hosts:
        hosts.append({"label": fallback, "url": fallback})
    return hosts


def _get_corpora():
    from .models import Corpus

    return Corpus.query.order_by(Corpus.fullname.asc()).all()

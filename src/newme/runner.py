from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from typing import Any


def format_dialogue(utterances) -> str:
    return "\n".join(f"[{u.position}] {u.author}: {u.text}" for u in utterances)


def assemble_prompt_text(
    prompt, dialogue_text: str, previous_output: str, user_settings, regex_candidates: str = ""
) -> str:
    parts = []

    if user_settings and user_settings.global_template:
        parts.append(user_settings.global_template.strip())

    parts.append(prompt.prompt_text.strip())

    text = "\n\n".join(p for p in parts if p)
    text = text.replace("{dialogue}", dialogue_text)
    text = text.replace("{previous_output}", previous_output)
    text = text.replace("{regex_candidates}", regex_candidates)

    if prompt.output_format == "simplified":
        appendix = (
            user_settings.effective_simplified_appendix
            if user_settings
            else _simplified_default()
        )
    elif prompt.output_format == "detailed":
        appendix = (
            user_settings.effective_detailed_appendix
            if user_settings
            else _detailed_default()
        )
    else:
        appendix = ""

    if appendix:
        text = text + "\n\n" + appendix.strip()

    return text


def _simplified_default() -> str:
    from .models.experiment import SIMPLIFIED_APPENDIX_DEFAULT
    return SIMPLIFIED_APPENDIX_DEFAULT


def _detailed_default() -> str:
    from .models.experiment import DETAILED_APPENDIX_DEFAULT
    return DETAILED_APPENDIX_DEFAULT


def _get_output_schema(output_format: str) -> dict | None:
    if output_format == "simplified":
        return {
            "type": "object",
            "properties": {
                "hits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "dialogue_id": {"type": "string"},
                            "start_index": {"type": "integer"},
                            "end_index": {"type": "integer"},
                            "label": {"type": "string"},
                        },
                        "required": ["dialogue_id", "start_index", "end_index", "label"],
                    },
                }
            },
            "required": ["hits"],
        }
    if output_format == "detailed":
        return {
            "type": "object",
            "properties": {
                "hits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "dialogue_id": {"type": "string"},
                            "start_index": {"type": "integer"},
                            "end_index": {"type": "integer"},
                            "start_offset": {"type": "integer"},
                            "end_offset": {"type": "integer"},
                            "label": {"type": "string"},
                            "excerpt": {"type": "string"},
                            "wmn_type": {"type": "string"},
                        },
                        "required": [
                            "dialogue_id", "start_index", "end_index",
                            "start_offset", "end_offset", "label", "excerpt",
                        ],
                    },
                }
            },
            "required": ["hits"],
        }
    # "regex" and None: no LLM schema
    return None


_VALID_LABEL_NAMES = {"Trigger", "Indicator", "Negotiation"}

# Matches BNC-style spaced contractions: "what 's" → "what's", "do n't" → "don't"
_CONTRACTION_RE = re.compile(r"(\w) (n't|'s|'re|'m|'ve|'ll|'d)\b")


def _normalize_transcript(text: str) -> str:
    return _CONTRACTION_RE.sub(r"\1\2", text)


def _run_regex(
    patterns_json: str,
    utterances: list,
    dialogue_external_id: str,
) -> list[dict[str, Any]]:
    """Run regex patterns against dialogue utterances.

    prompt_text must be a JSON array of pattern objects:
      [{"label": "Indicator", "pattern": "what do you mean", "flags": "i"}, ...]

    Supported labels: Trigger, Indicator, Negotiation.
    Each match becomes an annotated-style hit.
    """

    try:
        rules = json.loads(patterns_json)
    except (json.JSONDecodeError, TypeError):
        raise ValueError(
            "Regex prompt_text must be valid JSON — an array of "
            '{\"label\": ..., \"pattern\": ..., \"flags\": \"i\"} objects.'
        )

    if not isinstance(rules, list):
        raise ValueError("Regex patterns must be a JSON array.")

    compiled: list[tuple[str, re.Pattern]] = []
    for rule in rules:
        label = rule.get("label", "")
        if label not in _VALID_LABEL_NAMES:
            continue
        flags_str = rule.get("flags", "")
        re_flags = 0
        if "i" in flags_str:
            re_flags |= re.IGNORECASE
        if "s" in flags_str:
            re_flags |= re.DOTALL
        compiled.append((label, re.compile(rule["pattern"], re_flags)))

    hits: list[dict[str, Any]] = []
    for idx, utt in enumerate(utterances):
        original = utt.text
        normalized = _normalize_transcript(original)
        matched_labels: set[str] = set()
        for label, pattern in compiled:
            if label in matched_labels:
                continue
            if pattern.search(normalized):
                matched_labels.add(label)
                hits.append({
                    "dialogue_id": dialogue_external_id,
                    "start_index": idx,
                    "end_index": idx,
                    "start_offset": 0,
                    "end_offset": len(original),
                    "label": label,
                    "excerpt": original,
                })

    return hits


def _get_regex_candidates(experiment, dialogue_external_id: str, utterances: list) -> str:
    from .models.experiment import RegexRun, RegexRunResult

    if not experiment.regex_enabled:
        return ""

    regex_run = (
        RegexRun.query.filter_by(experiment_id=experiment.id, status="complete")
        .order_by(RegexRun.completed_at.desc())
        .first()
    )
    if not regex_run:
        return ""

    result = RegexRunResult.query.filter_by(
        regex_run_id=regex_run.id,
        dialogue_external_id=dialogue_external_id,
    ).first()

    if not result or not result.output:
        return ""

    utt_map = {u.position: u for u in utterances}
    seen: set[int] = set()
    hit_lines = []
    for hit in result.output:
        idx = hit.get("start_index")
        if idx is None or idx in seen:
            continue
        seen.add(idx)
        utt = utt_map.get(idx)
        if utt:
            hit_lines.append(f"[{idx}] {utt.author}: {utt.text}  [{hit.get('label', '')}]")

    if not hit_lines:
        return ""

    lines = [
        "The following utterances were flagged by a regex pre-filter as likely "
        "Indicators (high recall, low precision — treat as hints, not ground truth):",
        *hit_lines,
        "These are starting points. Look for Indicators the pre-filter may have "
        "missed, and do not assume every flagged utterance is genuinely an Indicator.",
    ]
    return "\n".join(lines) + "\n"


def _get_previous_output(prompt, dialogue_external_id: str) -> str:
    from .models.experiment import Prompt, Run, RunResult

    if prompt.position <= 1:
        return ""

    prev_prompt = Prompt.query.filter_by(
        experiment_id=prompt.experiment_id,
        position=prompt.position - 1,
    ).first()
    if not prev_prompt:
        return ""

    prev_run = (
        Run.query.filter_by(
            experiment_id=prompt.experiment_id,
            prompt_id=prev_prompt.id,
            status="complete",
        )
        .order_by(Run.completed_at.desc())
        .first()
    )
    if not prev_run:
        return ""

    prev_result = RunResult.query.filter_by(
        run_id=prev_run.id,
        dialogue_external_id=dialogue_external_id,
    ).first()

    if not prev_result or prev_result.output is None:
        return ""

    return json.dumps(prev_result.output)


def execute_run(run_id: int, app) -> None:
    thread = threading.Thread(target=_run_worker, args=(run_id, app), daemon=True)
    thread.start()


def execute_regex_run(regex_run_id: int, app) -> None:
    thread = threading.Thread(target=_regex_run_worker, args=(regex_run_id, app), daemon=True)
    thread.start()


def _regex_run_worker(regex_run_id: int, app) -> None:
    with app.app_context():
        from .extensions import db
        from .models import Corpus, Dialogue, Utterance
        from .models.experiment import (
            REGEX_PATTERNS_DEFAULT,
            ExperimentDialogue,
            RegexRun,
            RegexRunResult,
            UserSettings,
        )

        regex_run = db.session.get(RegexRun, regex_run_id)
        if not regex_run:
            return

        regex_run.status = "running"
        regex_run.started_at = datetime.now(timezone.utc)
        db.session.commit()

        try:
            experiment = regex_run.experiment
            user_settings = db.session.get(UserSettings, experiment.user_email)

            if experiment.regex_patterns is not None:
                patterns_json = experiment.regex_patterns
            elif user_settings is not None:
                patterns_json = user_settings.effective_regex_patterns
            else:
                patterns_json = REGEX_PATTERNS_DEFAULT

            dialogues = ExperimentDialogue.query.filter_by(experiment_id=experiment.id).all()
            regex_run.total_count = len(dialogues)
            db.session.commit()

            for exp_dialogue in dialogues:
                output = None
                error_msg = None

                try:
                    dialogue_obj = (
                        db.session.query(Dialogue)
                        .join(Corpus, Dialogue.corpus_id == Corpus.id)
                        .filter(
                            Corpus.codename == exp_dialogue.corpus_codename,
                            Dialogue.external_id == exp_dialogue.dialogue_external_id,
                        )
                        .first()
                    )
                    if dialogue_obj is None:
                        raise ValueError(
                            f"Dialogue {exp_dialogue.dialogue_external_id!r} not found"
                        )

                    utterances = (
                        Utterance.query.filter_by(dialogue_id=dialogue_obj.id)
                        .order_by(Utterance.position.asc())
                        .all()
                    )
                    output = _run_regex(
                        patterns_json, utterances, exp_dialogue.dialogue_external_id
                    )

                except Exception as exc:
                    error_msg = str(exc)

                db.session.add(RegexRunResult(
                    regex_run_id=regex_run.id,
                    dialogue_external_id=exp_dialogue.dialogue_external_id,
                    corpus_codename=exp_dialogue.corpus_codename,
                    output=output,
                    error=error_msg,
                ))
                regex_run.processed_count += 1
                db.session.commit()

            regex_run.status = "complete"
            regex_run.completed_at = datetime.now(timezone.utc)
            db.session.commit()

        except Exception as exc:
            regex_run = db.session.get(RegexRun, regex_run_id)
            if regex_run:
                regex_run.status = "error"
                regex_run.error_message = str(exc)
                regex_run.completed_at = datetime.now(timezone.utc)
                db.session.commit()


def _run_worker(run_id: int, app) -> None:
    with app.app_context():
        from .extensions import db
        from .models import Corpus, Dialogue, Utterance
        from .models.experiment import ExperimentDialogue, Prompt, Run, RunResult, UserSettings
        from .ollama_client import get_client

        run = db.session.get(Run, run_id)
        if not run:
            return

        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        db.session.commit()

        _unload_client = None
        _unload_model = None

        try:
            prompt = db.session.get(Prompt, run.prompt_id)
            experiment = prompt.experiment
            user_settings = db.session.get(UserSettings, experiment.user_email)

            dialogues = ExperimentDialogue.query.filter_by(experiment_id=experiment.id).all()
            run.total_count = len(dialogues)
            db.session.commit()

            is_regex = (prompt.output_format == "regex")
            host = prompt.host or "http://127.0.0.1:11434"
            client = None if is_regex else get_client(host)
            if client is not None:
                _unload_client = client
                _unload_model = prompt.model
            schema = _get_output_schema(prompt.output_format)

            if not is_regex:
                from .ollama_client import list_models
                available = list_models(host)
                if not available:
                    raise RuntimeError(
                        f"Could not connect to Ollama at {host}. Is it running?"
                    )
                if prompt.model not in available:
                    raise RuntimeError(
                        f"Model '{prompt.model}' is not available at {host}. "
                        f"Available: {', '.join(available)}"
                    )

            for exp_dialogue in dialogues:
                output = None
                raw_response = None
                error_msg = None

                try:
                    dialogue_obj = (
                        db.session.query(Dialogue)
                        .join(Corpus, Dialogue.corpus_id == Corpus.id)
                        .filter(
                            Corpus.codename == exp_dialogue.corpus_codename,
                            Dialogue.external_id == exp_dialogue.dialogue_external_id,
                        )
                        .first()
                    )
                    if dialogue_obj is None:
                        raise ValueError(
                            f"Dialogue {exp_dialogue.dialogue_external_id!r} not found in database"
                        )

                    utterances = (
                        Utterance.query.filter_by(dialogue_id=dialogue_obj.id)
                        .order_by(Utterance.position.asc())
                        .all()
                    )

                    if is_regex:
                        output = _run_regex(
                            prompt.prompt_text,
                            utterances,
                            exp_dialogue.dialogue_external_id,
                        )
                    else:
                        dialogue_text = format_dialogue(utterances)
                        previous_output = _get_previous_output(
                            prompt, exp_dialogue.dialogue_external_id
                        )
                        regex_candidates = _get_regex_candidates(
                            experiment, exp_dialogue.dialogue_external_id, utterances
                        )
                        prompt_text = assemble_prompt_text(
                            prompt, dialogue_text, previous_output, user_settings, regex_candidates
                        )

                        messages = []
                        if prompt.system_prompt:
                            messages.append({"role": "system", "content": prompt.system_prompt})
                        messages.append({"role": "user", "content": prompt_text})

                        chat_kwargs: dict = {
                            "model": prompt.model,
                            "messages": messages,
                        }
                        options: dict = {}
                        if prompt.temperature is not None:
                            options["temperature"] = prompt.temperature
                        if prompt.num_ctx is not None:
                            options["num_ctx"] = prompt.num_ctx
                        if options:
                            chat_kwargs["options"] = options
                        if schema:
                            chat_kwargs["format"] = schema

                        response = client.chat(**chat_kwargs)
                        raw_response = response.message.content

                        if schema:
                            parsed = json.loads(raw_response)
                            hits = parsed.get("hits", [])
                            if prompt.output_format == "simplified" and hits:
                                utt_map = {
                                    u.position: f"[{u.position}] {u.author}: {u.text}"
                                    for u in utterances
                                }
                                for hit in hits:
                                    start = hit.get("start_index")
                                    end = hit.get("end_index")
                                    if start is not None:
                                        end = end if end is not None else start
                                        texts = [
                                            utt_map[i]
                                            for i in range(start, end + 1)
                                            if i in utt_map
                                        ]
                                        hit["excerpt"] = "\n".join(texts)
                            output = hits
                        else:
                            output = raw_response

                except Exception as exc:
                    error_msg = str(exc)

                db.session.add(RunResult(
                    run_id=run.id,
                    dialogue_external_id=exp_dialogue.dialogue_external_id,
                    corpus_codename=exp_dialogue.corpus_codename,
                    output=output,
                    raw_response=raw_response,
                    error=error_msg,
                ))
                run.processed_count += 1
                db.session.commit()

            run.status = "complete"
            run.completed_at = datetime.now(timezone.utc)
            db.session.commit()

        except Exception as exc:
            run = db.session.get(Run, run_id)
            if run:
                run.status = "error"
                run.error_message = str(exc)
                run.completed_at = datetime.now(timezone.utc)
                db.session.commit()
        finally:
            if _unload_client is not None and _unload_model:
                try:
                    _unload_client.generate(model=_unload_model, prompt="", keep_alive=0)
                except Exception:
                    pass

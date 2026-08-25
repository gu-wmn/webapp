from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any

# Maps an in-progress LLM run to the Ollama client it's currently using, so an
# abort request from a different thread can close just that run's own
# connection — never a different run's, even one hitting the same host.
_active_clients_lock = threading.Lock()
_active_clients: dict[int, Any] = {}


def request_ollama_abort(run_id: int) -> bool:
    """Best-effort: close the given run's own Ollama connection so a blocked
    request unblocks instead of running to completion with nobody listening.

    Only ever touches the client registered for this specific run_id — other
    runs, including ones talking to the same host, are unaffected since each
    run owns its own ollama.Client / connection, never a shared one.

    Returns True if a client was found for this run (closing it is fire-and-
    forget; if it fails, the caller still has the DB-level abort as the
    reliable fallback).
    """
    with _active_clients_lock:
        client = _active_clients.get(run_id)
    if client is None:
        return False
    try:
        client._client.close()
    except Exception:
        pass
    return True


def format_dialogue(utterances, dialogue_external_id: str, corpus_codename: str) -> str:
    payload = {
        "dialogue": {
            "dialogue_id": dialogue_external_id,
            "corpus_codename": corpus_codename,
            "utterances": [
                {
                    "utterance_index": u.position,
                    "speaker": u.author,
                    "text": u.text,
                }
                for u in utterances
            ],
        }
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def assemble_prompt_text(
    prompt,
    dialogue_text: str,
    previous_output: str,
    user_settings,
    regex_candidates: str = "",
) -> str:
    parts = []

    if prompt.include_global_template:
        template = (
            user_settings.effective_global_template
            if user_settings
            else _global_template_default()
        )
        if template:
            parts.append(template.strip())

    if prompt.position <= 1 or prompt.include_dialogue:
        header = (
            user_settings.effective_dialogue_input_instructions
            if user_settings
            else _dialogue_input_default()
        )
        parts.append(header.strip() + "\n" + dialogue_text)

    if prompt.include_regex_candidates:
        header = (
            user_settings.effective_regex_input_instructions
            if user_settings
            else _regex_input_default()
        )
        parts.append(header.strip() + "\n" + regex_candidates)

    if prompt.position > 1:
        header = (
            user_settings.effective_previous_output_instructions
            if user_settings
            else _previous_output_default()
        )
        parts.append(header.strip() + "\n" + previous_output)

    parts.append(prompt.prompt_text.strip())

    text = "\n\n".join(p for p in parts if p)

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
        appendix = (
            user_settings.effective_free_text_appendix
            if user_settings
            else ""
        )

    if appendix:
        text = text + "\n\n" + appendix.strip()

    return text


# Chars-per-token estimate for English text. A flat 4 is the usual rule of
# thumb for prose, but a large share of every prompt here is JSON (the
# serialized dialogue, which dominates character count on longer dialogues) —
# punctuation, quotes, and repeated keys tokenize less efficiently than prose,
# commonly closer to 3.3-3.7 chars/token. Using 3.5 errs toward overestimating
# the token count, which is the safe direction: it costs a little unused
# context, where underestimating costs truncated dialogue.
_CHARS_PER_TOKEN = 3.5
# Headroom reserved for the model's response (the JSON hits array, plus
# whatever reasoning a model does before it) — the app never sets a
# num_predict cap, so this is the only thing keeping the response from
# crowding out context that's needed for the input. Scales with input size
# (longer dialogues tend to surface more candidate WMNs, so more hits to
# report) on top of a flat floor, capped so it doesn't balloon on huge inputs.
_OUTPUT_TOKEN_RESERVE_FLOOR = 2048
_OUTPUT_TOKEN_RESERVE_FRACTION = 0.08
_OUTPUT_TOKEN_RESERVE_CAP = 8192
_MIN_NUM_CTX = 2048
# Rounding the raw estimate straight up to the next power of two gives a
# sawtooth headroom pattern: generous right after crossing a boundary, but
# shrinking toward almost nothing right before the next one — a prompt that
# needs 131,000 tokens gets the same 131,072 window as one that needs 65,537,
# leaving it almost no margin for estimation error. Padding the estimate by
# this factor before rounding guarantees a floor on headroom everywhere in a
# tier, not just at its start — 1.42x reproduces a real-world scaling table
# (16K/32K/64K/128K/256K windows at the 8K/20K/45K/90K/... input marks) almost
# exactly. No upper cap: correctness (not truncating) matters more than
# bounding worst-case memory use, so a large enough dialogue keeps doubling.
_SAFETY_MARGIN_MULTIPLIER = 1.42

# This is span extraction/classification against a fixed schema, not creative
# generation — there's a defensible "right" answer for a given dialogue, so
# the model should commit to its best read rather than sample around it. Not
# 0 (fully greedy): a little randomness avoids the occasional repetition loop
# greedy decoding can fall into on longer dialogues, at negligible cost to
# run-to-run consistency.
DEFAULT_TEMPERATURE = 0.1


def _adaptive_num_ctx(prompt_length: int) -> int:
    """Size the context window to an assembled prompt's length rather than
    trusting whatever a model's own default happens to be — Ollama's
    defaults (often 2048-4096) silently truncate long dialogues instead of
    erroring, which would quietly corrupt results. Only used when a prompt
    doesn't set its own num_ctx override.

    Takes a length, not the prompt text itself: llama.cpp fixes num_ctx at
    model-load time, so a value that varies per request forces Ollama to
    reload the entire model before serving each one — callers must compute
    this once per run (from the largest dialogue in it) and reuse it for
    every dialogue, never call it per-dialogue.

    The padded total is rounded up to the next power of two, matching how
    context windows are conventionally sized. No upper bound — only a floor
    for degenerate (near-empty) prompts.
    """
    input_tokens = -int(-prompt_length // _CHARS_PER_TOKEN)  # ceil division
    output_reserve = min(
        _OUTPUT_TOKEN_RESERVE_CAP,
        max(_OUTPUT_TOKEN_RESERVE_FLOOR, int(input_tokens * _OUTPUT_TOKEN_RESERVE_FRACTION)),
    )
    needed = (input_tokens + output_reserve) * _SAFETY_MARGIN_MULTIPLIER
    num_ctx = 1
    while num_ctx < needed:
        num_ctx *= 2
    return max(_MIN_NUM_CTX, num_ctx)


def _simplified_default() -> str:
    from .models.experiment import SIMPLIFIED_APPENDIX_DEFAULT
    return SIMPLIFIED_APPENDIX_DEFAULT


def _global_template_default() -> str:
    from .models.experiment import GLOBAL_TEMPLATE_DEFAULT
    return GLOBAL_TEMPLATE_DEFAULT


def _detailed_default() -> str:
    from .models.experiment import DETAILED_APPENDIX_DEFAULT
    return DETAILED_APPENDIX_DEFAULT


def _dialogue_input_default() -> str:
    from .models.experiment import DIALOGUE_INPUT_INSTRUCTIONS_DEFAULT
    return DIALOGUE_INPUT_INSTRUCTIONS_DEFAULT


def _regex_input_default() -> str:
    from .models.experiment import REGEX_INPUT_INSTRUCTIONS_DEFAULT
    return REGEX_INPUT_INSTRUCTIONS_DEFAULT


def _previous_output_default() -> str:
    from .models.experiment import PREVIOUS_OUTPUT_INSTRUCTIONS_DEFAULT
    return PREVIOUS_OUTPUT_INSTRUCTIONS_DEFAULT


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
                            "utterance_start_index": {"type": "integer"},
                            "utterance_end_index": {"type": "integer"},
                            "label": {"type": "string"},
                            "quote": {"type": "string"},
                        },
                        "required": [
                            "dialogue_id",
                            "utterance_start_index",
                            "utterance_end_index",
                            "label",
                            "quote",
                        ],
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
                            "utterance_start_index": {"type": "integer"},
                            "utterance_end_index": {"type": "integer"},
                            "label": {"type": "string"},
                            "quote": {"type": "string"},
                            "wmn_type": {"type": "string"},
                            "wmn_group": {"type": "integer"},
                        },
                        "required": [
                            "dialogue_id",
                            "utterance_start_index",
                            "utterance_end_index",
                            "label",
                            "quote",
                            "wmn_type",
                            "wmn_group",
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


def _serialize_hits_payload(hits: list[dict[str, Any]]) -> str:
    return json.dumps({"hits": hits}, ensure_ascii=False, indent=2)


def _filter_valid_hits(hits: list, utterances: list) -> list:
    """Drop hits whose quote cannot be located in their stated utterance range."""
    from .models.experiment import MULTI_UTTERANCE_QUOTE_SEPARATOR

    valid = []
    n = len(utterances)
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        try:
            start = int(hit.get("utterance_start_index") or hit.get("start_index") or 0)
            end = int(hit.get("utterance_end_index") or hit.get("end_index") or start)
        except (TypeError, ValueError):
            continue
        quote = str(hit.get("quote") or hit.get("excerpt") or "")
        if not quote or start < 0 or end < start or end >= n:
            continue

        range_text = "\n".join(utterances[i].text for i in range(start, end + 1))
        if quote in range_text:
            valid.append(hit)
            continue

        # Multi-utterance hits may quote only the boundary utterances, joined by
        # MULTI_UTTERANCE_QUOTE_SEPARATOR, instead of the full verbatim span.
        if start != end and MULTI_UTTERANCE_QUOTE_SEPARATOR in quote:
            head, _, tail = quote.partition(MULTI_UTTERANCE_QUOTE_SEPARATOR)
            head, tail = head.strip(), tail.strip()
            head_ok = (not head) or (head in utterances[start].text)
            tail_ok = (not tail) or (tail in utterances[end].text)
            if head_ok and tail_ok:
                valid.append(hit)

    return valid


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

    compiled: list[tuple[str, str, re.Pattern]] = []
    for rule in rules:
        label = rule.get("label", "")
        if label not in _VALID_LABEL_NAMES:
            continue
        pattern_text = str(rule.get("pattern") or "")
        flags_str = rule.get("flags", "")
        re_flags = 0
        if "i" in flags_str:
            re_flags |= re.IGNORECASE
        if "s" in flags_str:
            re_flags |= re.DOTALL
        compiled.append((label, pattern_text, re.compile(pattern_text, re_flags)))

    hits: list[dict[str, Any]] = []
    for idx, utt in enumerate(utterances):
        original = utt.text
        normalized = _normalize_transcript(original)
        matched_labels: set[str] = set()
        for label, pattern_text, pattern in compiled:
            if label in matched_labels:
                continue
            match = pattern.search(original)
            if match is None:
                match = pattern.search(normalized)
            if match is not None:
                matched_labels.add(label)
                if pattern.search(original):
                    start_index = match.start()
                    end_index = match.end()
                    quote = original[start_index:end_index]
                else:
                    start_index = 0
                    end_index = len(original)
                    quote = original
                hits.append({
                    "utterance_start_index": idx,
                    "utterance_end_index": idx,
                    "char_start_index": start_index,
                    "char_end_index": end_index,
                    "label": label,
                    "quote": quote,
                    "matched_by": [pattern_text],
                })

    return hits


def _get_regex_candidates(experiment, dialogue_external_id: str, utterances: list) -> str:
    from .models.experiment import RegexRun, RegexRunResult

    regex_run = (
        RegexRun.query.filter_by(experiment_id=experiment.id, status="complete")
        .order_by(RegexRun.completed_at.desc())
        .first()
    )
    if not regex_run:
        return _serialize_hits_payload([])

    result = RegexRunResult.query.filter_by(
        regex_run_id=regex_run.id,
        dialogue_external_id=dialogue_external_id,
    ).first()

    if not result or not result.output:
        return _serialize_hits_payload([])

    return _serialize_hits_payload(result.output if isinstance(result.output, list) else [])


def _get_previous_output(prompt, dialogue_external_id: str, utterances: list) -> str:
    from .models.experiment import Prompt, Run, RunResult

    if prompt.position <= 1:
        return _serialize_hits_payload([])

    prev_prompt = Prompt.query.filter_by(
        experiment_id=prompt.experiment_id,
        position=prompt.position - 1,
    ).first()
    if not prev_prompt:
        return _serialize_hits_payload([])

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
        return _serialize_hits_payload([])

    prev_result = RunResult.query.filter_by(
        run_id=prev_run.id,
        dialogue_external_id=dialogue_external_id,
    ).first()

    if not prev_result or prev_result.output is None:
        return _serialize_hits_payload([])

    raw_hits = prev_result.output if isinstance(prev_result.output, list) else []
    return _serialize_hits_payload(_filter_valid_hits(raw_hits, utterances))


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

            if user_settings is not None:
                patterns_json = user_settings.effective_regex_patterns
            else:
                patterns_json = REGEX_PATTERNS_DEFAULT

            dialogues = ExperimentDialogue.query.filter_by(experiment_id=experiment.id).all()
            regex_run.total_count = len(dialogues)
            db.session.commit()

            for exp_dialogue in dialogues:
                db.session.refresh(regex_run)
                if regex_run.status not in ("pending", "running"):
                    return

                output = None
                error_msg = None
                char_count = None
                started = time.monotonic()

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
                    char_count = sum(len(u.text) for u in utterances)
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
                    dialogue_char_count=char_count,
                    duration_seconds=time.monotonic() - started,
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


def _chat_with_retry(client, chat_kwargs: dict, retries: int = 1, backoff_seconds: float = 2.0) -> str:
    """Call client.chat() with stream=True and return the fully accumulated
    response text, retrying once on a transport-level failure.

    Streams rather than making one blocking non-streaming call: a
    non-streaming request sits completely silent on the wire for the whole
    generation (we saw this take 10+ minutes on real dialogues), and that
    silence is exactly what idle-connection timeouts in NAT gateways,
    firewalls, and load balancers tend to kill — Ollama finishes and tries
    to send the response into a connection that's already been dropped
    somewhere on the path, and it never arrives. Streaming keeps the
    connection actively transmitting chunks throughout, which should avoid
    that failure mode, and also means ollama_client's read timeout (the gap
    between chunks, not the whole response) becomes a meaningful signal
    instead of something that has to tolerate an entire generation's worth
    of silence.

    A dropped or timed-out connection is often transient — worth one more
    real attempt at getting results before giving up on this dialogue
    entirely. Only transport errors are retried; a bad model name, a schema
    violation, etc. would just fail identically again, so those propagate
    immediately.
    """
    import httpx

    attempt = 0
    while True:
        try:
            chunks: list[str] = []
            for part in client.chat(**chat_kwargs, stream=True):
                if part.message and part.message.content:
                    chunks.append(part.message.content)
            return "".join(chunks)
        except httpx.TransportError:
            if attempt >= retries:
                raise
            attempt += 1
            time.sleep(backoff_seconds)


def compute_experiment_char_count(experiment) -> int:
    """Total character count across every dialogue in the experiment's sample.

    Computed once when a run starts, so there's a fixed total to weigh
    completed-so-far characters against for an ETA — including before any
    dialogue in the run has actually finished.
    """
    from .extensions import db
    from .models import Corpus, Dialogue, Utterance
    from .models.experiment import ExperimentDialogue

    dialogues = ExperimentDialogue.query.filter_by(experiment_id=experiment.id).all()
    total = 0
    for exp_dialogue in dialogues:
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
            continue
        char_count = (
            db.session.query(db.func.sum(db.func.length(Utterance.text)))
            .filter(Utterance.dialogue_id == dialogue_obj.id)
            .scalar()
        )
        total += char_count or 0
    return total


def estimate_run_eta_seconds(run, prompt) -> tuple[float | None, str | None]:
    """Rough ETA, in seconds, for the dialogues still left in a run.

    Prefers a seconds-per-character rate derived from dialogues already
    completed in this run, since that reflects whatever's actually
    happening on the host right now (host load, model, current context
    sizing). Falls back to this model's rate from past runs so there's
    still something to show before the first dialogue in this run
    finishes. Returns (seconds, source), where source is "current_run",
    "historical", or None if neither has enough data yet.
    """
    from .extensions import db
    from .models.experiment import Prompt, Run, RunResult

    current_seconds, current_chars = (
        db.session.query(
            db.func.sum(RunResult.duration_seconds),
            db.func.sum(RunResult.dialogue_char_count),
        )
        .filter(
            RunResult.run_id == run.id,
            RunResult.duration_seconds.isnot(None),
            RunResult.dialogue_char_count > 0,
        )
        .first()
    )
    current_chars = current_chars or 0

    if current_chars:
        rate = current_seconds / current_chars
        source = "current_run"
    else:
        hist_seconds, hist_chars = (
            db.session.query(
                db.func.sum(RunResult.duration_seconds),
                db.func.sum(RunResult.dialogue_char_count),
            )
            .join(Run, RunResult.run_id == Run.id)
            .join(Prompt, Run.prompt_id == Prompt.id)
            .filter(
                Prompt.model == prompt.model,
                RunResult.duration_seconds.isnot(None),
                RunResult.dialogue_char_count > 0,
            )
            .first()
        )
        if not hist_chars:
            return None, None
        rate = hist_seconds / hist_chars
        source = "historical"

    if run.total_char_count is None:
        return None, None

    remaining_chars = max(run.total_char_count - current_chars, 0)
    return rate * remaining_chars, source


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
                with _active_clients_lock:
                    _active_clients[run_id] = client
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

            import httpx

            if is_regex:
                # Pure CPU pattern matching, no network call to overlap —
                # nothing to gain from parallelizing this path.
                for exp_dialogue in dialogues:
                    db.session.refresh(run)
                    if run.status not in ("pending", "running"):
                        return

                    output = None
                    error_msg = None
                    char_count = None
                    started = time.monotonic()

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
                        char_count = sum(len(u.text) for u in utterances)
                        output = _run_regex(
                            prompt.prompt_text,
                            utterances,
                            exp_dialogue.dialogue_external_id,
                        )
                    except Exception as exc:
                        error_msg = str(exc)

                    db.session.add(RunResult(
                        run_id=run.id,
                        dialogue_external_id=exp_dialogue.dialogue_external_id,
                        corpus_codename=exp_dialogue.corpus_codename,
                        output=output,
                        raw_response=None,
                        error=error_msg,
                        dialogue_char_count=char_count,
                        duration_seconds=time.monotonic() - started,
                    ))
                    run.processed_count += 1
                    db.session.commit()

            else:
                # One dialogue in flight at a time, not all of them. Ollama
                # has no way to cancel a request it has already started, so
                # sending everything at once just means every dialogue ends
                # up queued on Ollama's own side instead of ours — aborting
                # doesn't dequeue anything there, so whatever's still queued
                # keeps grinding through regardless, and a fresh prompt
                # started right after an abort would have to wait behind all
                # of it anyway. One at a time means an abort only ever has to
                # wait out the single dialogue Ollama happens to be working
                # on right now, not dozens.
                for exp_dialogue in dialogues:
                    db.session.refresh(run)
                    if run.status not in ("pending", "running"):
                        return

                    output = None
                    raw_response = None
                    error_msg = None
                    char_count = None
                    started = time.monotonic()

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
                        char_count = sum(len(u.text) for u in utterances)

                        dialogue_text = format_dialogue(
                            utterances,
                            exp_dialogue.dialogue_external_id,
                            exp_dialogue.corpus_codename,
                        )
                        previous_output = _get_previous_output(
                            prompt, exp_dialogue.dialogue_external_id, utterances
                        )
                        regex_candidates = _get_regex_candidates(
                            experiment, exp_dialogue.dialogue_external_id, utterances
                        )
                        prompt_text = assemble_prompt_text(
                            prompt,
                            dialogue_text,
                            previous_output,
                            user_settings,
                            regex_candidates,
                        )

                        chat_kwargs: dict = {
                            "model": prompt.model,
                            "messages": [{"role": "user", "content": prompt_text}],
                            "options": {
                                "num_ctx": (
                                    prompt.num_ctx if prompt.num_ctx is not None
                                    else _adaptive_num_ctx(len(prompt_text))
                                ),
                                "temperature": (
                                    prompt.temperature if prompt.temperature is not None
                                    else DEFAULT_TEMPERATURE
                                ),
                            },
                        }
                        if schema:
                            chat_kwargs["format"] = schema

                        # print(flush=True), not app.logger: Flask's default
                        # logger filters INFO below its effective level
                        # unless something has explicitly configured it,
                        # which nothing here does — a plain flushed print is
                        # guaranteed to reach the container's captured
                        # stdout regardless.
                        print(
                            f"run {run.id}: dispatching dialogue "
                            f"{exp_dialogue.dialogue_external_id} "
                            f"(num_ctx={chat_kwargs['options']['num_ctx']})",
                            flush=True,
                        )
                        raw_response = _chat_with_retry(client, chat_kwargs)

                        if schema:
                            parsed = json.loads(raw_response)
                            output = parsed.get("hits", [])
                        else:
                            output = raw_response

                    except httpx.TransportError as exc:
                        # The only case that's actually a connectivity
                        # problem — surfaced in the "Running" banner so
                        # it's visible before the run even finishes.
                        error_msg = f"Lost connection to Ollama host at {host}: {exc}"
                        run.last_error = error_msg
                    except json.JSONDecodeError as exc:
                        # Ollama responded — this isn't a connection issue. An
                        # empty raw_response (the common cause of this exact
                        # error) usually means the model ran out of context
                        # budget before producing any output.
                        preview = (raw_response or "")[:200]
                        error_msg = (
                            f"Model output was not valid JSON ({exc}). "
                            f"Raw response: {preview!r}"
                        )
                        run.last_error = None
                    except Exception as exc:
                        # Also not a connection issue — some other failure
                        # processing this one dialogue.
                        error_msg = str(exc)
                        run.last_error = None
                    else:
                        run.last_error = None

                    duration = time.monotonic() - started
                    db.session.add(RunResult(
                        run_id=run.id,
                        dialogue_external_id=exp_dialogue.dialogue_external_id,
                        corpus_codename=exp_dialogue.corpus_codename,
                        output=output,
                        raw_response=raw_response,
                        error=error_msg,
                        dialogue_char_count=char_count,
                        duration_seconds=duration,
                    ))
                    run.processed_count += 1
                    db.session.commit()
                    print(
                        f"run {run.id}: recorded dialogue "
                        f"{exp_dialogue.dialogue_external_id} "
                        f"({run.processed_count}/{run.total_count} done, "
                        f"took {duration:.1f}s)"
                        + (f" — error: {error_msg}" if error_msg else ""),
                        flush=True,
                    )

            run.status = "complete"
            run.completed_at = datetime.now(timezone.utc)
            run.last_error = None
            db.session.commit()

        except Exception as exc:
            run = db.session.get(Run, run_id)
            if run:
                run.status = "error"
                run.error_message = str(exc)
                run.last_error = None
                run.completed_at = datetime.now(timezone.utc)
                db.session.commit()
        finally:
            with _active_clients_lock:
                _active_clients.pop(run_id, None)
            if _unload_client is not None and _unload_model:
                try:
                    _unload_client.generate(model=_unload_model, prompt="", keep_alive=0)
                except Exception:
                    pass

from __future__ import annotations

import functools
import random
from datetime import datetime, timezone
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
    GLOBAL_TEMPLATE_DEFAULT,
    REGEX_FORMAT_HELP,
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


def _utterance_range_text(
    utterances: list[dict[str, str]],
    start_index: int,
    end_index: int,
) -> str:
    return "\n".join(utterances[idx]["text"] for idx in range(start_index, end_index + 1))


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

        if output_format == "simplified":
            utterance_text = _utterance_range_text(utterances, start_index, end_index)
            if quote not in utterance_text:
                issues.append({
                    "severity": "warning",
                    "message": (
                        f"Hit {hit_index} quote does not occur in the spanned utterance text: "
                        f"{quote!r}"
                    ),
                })
            continue

        start_offset = _hit_int(hit, "char_start_index", "start_offset")
        end_offset = _hit_int(hit, "char_end_index", "end_offset")
        if start_offset is None or end_offset is None:
            issues.append({
                "severity": "error",
                "message": f"Hit {hit_index} is missing char span indices.",
            })
            continue
        if start_offset < 0 or end_offset < start_offset:
            issues.append({
                "severity": "error",
                "message": f"Hit {hit_index} has an invalid char span.",
            })
            continue
        if start_offset > len(utterances[start_index]["text"]) or end_offset > len(utterances[end_index]["text"]):
            issues.append({
                "severity": "error",
                "message": f"Hit {hit_index} char span points outside the utterance text.",
            })
            continue

        span_text = _span_text_from_utterances(
            utterances,
            start_index,
            end_index,
            start_offset,
            end_offset,
        )
        if quote != span_text:
            issues.append({
                "severity": "warning",
                "message": (
                    f"Hit {hit_index} quote does not exactly match the indexed span. "
                    f"Quote: {quote!r} Span: {span_text!r}"
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
        if start_index < 0 or end_index < 0:
            continue
        if end_index < start_index:
            continue
        if start_index >= len(utterances) or end_index >= len(utterances):
            continue

        start_offset = _hit_int(hit, "char_start_index", "start_offset")
        end_offset = _hit_int(hit, "char_end_index", "end_offset")
        if start_offset is None:
            start_offset = 0
        if end_offset is None:
            end_offset = len(utterances[end_index]["text"])

        excerpt = _hit_text(hit, "quote", "excerpt")

        labels.append(
            {
                "name": name,
                "utterance_start_index": start_index,
                "utterance_end_index": end_index,
                "char_start_index": start_offset,
                "char_end_index": end_offset,
                "quote": excerpt,
                "wmn_type": str(hit.get("wmn_type") or "").strip(),
            }
        )

    return labels


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


@bp.get("/settings")
@login_required
def settings():
    user = session["user"]
    user_settings = db.session.get(UserSettings, user)
    return render_template(
        "portal/settings.html",
        user=user,
        us=user_settings,
        global_template_default=GLOBAL_TEMPLATE_DEFAULT,
        simplified_appendix_default=SIMPLIFIED_APPENDIX_DEFAULT,
        detailed_appendix_default=DETAILED_APPENDIX_DEFAULT,
        regex_patterns_default=REGEX_PATTERNS_DEFAULT,
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
    user_settings.simplified_appendix = (request.form.get("simplified_appendix") or "").strip() or None
    user_settings.detailed_appendix = (request.form.get("detailed_appendix") or "").strip() or None
    user_settings.regex_patterns = (request.form.get("regex_patterns") or "").strip() or None
    db.session.commit()
    return redirect(url_for("portal.settings"))


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

    return render_template(
        "portal/experiment.html",
        user=user,
        experiment=exp,
        user_settings=user_settings,
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
    user_settings = db.session.get(UserSettings, user)

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
            next_position = (
                db.session.query(db.func.max(Prompt.position))
                .filter_by(experiment_id=exp.id)
                .scalar()
                or 0
            ) + 1

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

            output_format = output_format_raw if output_format_raw in ("simplified", "detailed") else None

            db.session.add(Prompt(
                experiment_id=exp.id,
                position=next_position,
                name=name,
                host=host or None,
                model=model or "—",
                include_global_template=request.form.get("include_global_template") == "on",
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
        user_settings=user_settings,
        error=error,
    )


@bp.get("/experiments/<int:experiment_id>/prompts/<int:prompt_id>/edit")
@bp.post("/experiments/<int:experiment_id>/prompts/<int:prompt_id>/edit")
@login_required
def edit_prompt(experiment_id: int, prompt_id: int):
    user = session["user"]
    exp = Experiment.query.filter_by(id=experiment_id, user_email=user).first_or_404()
    prompt = Prompt.query.filter_by(id=prompt_id, experiment_id=exp.id).first_or_404()
    user_settings = db.session.get(UserSettings, user)

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
            prompt.system_prompt = None

            output_format = output_format_raw if output_format_raw in ("simplified", "detailed") else None
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
        error=error,
    )


@bp.post("/experiments/<int:experiment_id>/regex/toggle")
@login_required
def toggle_regex(experiment_id: int):
    user = session["user"]
    exp = Experiment.query.filter_by(id=experiment_id, user_email=user).first_or_404()
    exp.regex_enabled = not exp.regex_enabled
    db.session.commit()
    return redirect(url_for("portal.experiment", experiment_id=exp.id))


@bp.get("/experiments/<int:experiment_id>/regex/edit")
@bp.post("/experiments/<int:experiment_id>/regex/edit")
@login_required
def edit_regex(experiment_id: int):
    user = session["user"]
    exp = Experiment.query.filter_by(id=experiment_id, user_email=user).first_or_404()

    if request.method == "POST":
        exp.regex_patterns = (request.form.get("regex_patterns") or "").strip() or None
        db.session.commit()
        return redirect(url_for("portal.experiment", experiment_id=exp.id))

    user_settings = db.session.get(UserSettings, user)
    default = user_settings.effective_regex_patterns if user_settings else REGEX_PATTERNS_DEFAULT
    current = exp.regex_patterns if exp.regex_patterns is not None else default

    return render_template(
        "portal/edit_regex.html",
        user=user,
        experiment=exp,
        regex_patterns=current,
        regex_format_help=REGEX_FORMAT_HELP,
    )


@bp.post("/experiments/<int:experiment_id>/regex/run")
@login_required
def start_regex_run(experiment_id: int):
    from flask import current_app

    from .runner import execute_regex_run

    user = session["user"]
    exp = Experiment.query.filter_by(id=experiment_id, user_email=user).first_or_404()

    if not exp.regex_enabled:
        return jsonify({"error": "Regex pre-filter is not enabled."}), 400
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
    return jsonify({"run_id": regex_run.id})


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

    from .runner import execute_run

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
    )
    db.session.add(run)
    db.session.commit()

    execute_run(run.id, current_app._get_current_object())
    return jsonify({"run_id": run.id})


@bp.get("/api/runs/<int:run_id>/status")
@login_required
def run_status(run_id: int):
    run = db.session.get(Run, run_id)
    if run is None:
        return jsonify({"error": "Not found"}), 404
    Experiment.query.filter_by(
        id=run.experiment_id, user_email=session["user"]
    ).first_or_404()

    return jsonify({
        "run_id": run.id,
        "status": run.status,
        "processed_count": run.processed_count,
        "total_count": run.total_count,
        "error_message": run.error_message,
    })


def _human_utterance_set(sequences) -> set[int]:
    """Return the set of utterance indices covered by human annotation labels."""
    indices: set[int] = set()
    for seq in sequences:
        for label in seq.labels:
            if label.name not in _VALID_LABEL_NAMES:
                continue
            if label.start_index is None or label.end_index is None:
                continue
            for i in range(label.start_index, label.end_index + 1):
                indices.add(i)
    return indices


def _llm_utterance_set(result: RunResult) -> set[int]:
    """Return the set of utterance indices covered by LLM output hits."""
    if not isinstance(result.output, list):
        return set()
    indices: set[int] = set()
    for hit in result.output:
        if not isinstance(hit, dict):
            continue
        start = hit.get("utterance_start_index", hit.get("start_index"))
        end = hit.get("utterance_end_index", hit.get("end_index"))
        if start is None:
            continue
        end = end if end is not None else start
        for i in range(int(start), int(end) + 1):
            indices.add(i)
    return indices


def _compute_run_metrics(results: list[RunResult]) -> tuple[dict, dict[int, dict]]:
    """Compute utterance-level precision/recall/F1 for a run against human annotations.

    Returns:
        aggregate: dict with precision, recall, f1, tp, fp, fn, dialogues_with_hits,
                   dialogues_with_human, dialogues_both
        per_result: dict mapping result.id -> {matched, llm_count, human_count}
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

    tp = fp = fn = 0
    dialogues_with_hits = dialogues_with_human = dialogues_both = 0
    per_result: dict[int, dict] = {}

    for result in results:
        key = (result.corpus_codename, result.dialogue_external_id)
        human_seqs = seqs_by_dialogue.get(key, [])
        human_set = _human_utterance_set(human_seqs)
        llm_set = _llm_utterance_set(result)

        matched = len(human_set & llm_set)
        tp += matched
        fp += len(llm_set - human_set)
        fn += len(human_set - llm_set)

        if llm_set:
            dialogues_with_hits += 1
        if human_set:
            dialogues_with_human += 1
        if llm_set and human_set:
            dialogues_both += 1

        per_result[result.id] = {
            "matched": matched,
            "llm_count": len(llm_set),
            "human_count": len(human_set),
        }

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None

    aggregate = {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
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
        Run.query.filter_by(prompt_id=prompt.id, status="complete")
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

    run = (
        Run.query.filter_by(prompt_id=prompt.id, status="complete")
        .order_by(Run.completed_at.desc())
        .first_or_404()
    )
    result = RunResult.query.filter_by(id=result_id, run_id=run.id).first_or_404()

    utterances = _load_dialogue_utterances(result.corpus_codename, result.dialogue_external_id)
    llm_labels = _result_label_payload(result, utterances)
    llm_validation_issues = _validate_result_hits(
        result.output,
        utterances,
        output_format=prompt.output_format,
    )
    llm_utterances: list[dict[str, str]] = []
    llm_label_links: list[dict[str, str]] = []
    if utterances:
        llm_utterances, llm_label_links = _annotate_dialogue_utterances(
            utterances,
            llm_labels,
            anchor_prefix="llm-label",
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
    human_labels: list[dict[str, Any]] = []
    human_label_links: list[dict[str, str]] = []
    if selected_sequence is not None and utterances:
        human_labels = _sequence_label_payload(selected_sequence)
        human_utterances, human_label_links = _annotate_dialogue_utterances(
            utterances,
            human_labels,
            anchor_prefix="human-label",
        )

    llm_hit_count = len(llm_labels)

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
        human_sequences=human_sequences,
        selected_sequence=selected_sequence,
        human_utterances=human_utterances,
        human_labels=human_labels,
        human_label_links=human_label_links,
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

from __future__ import annotations

import functools
import random
from datetime import datetime, timezone

from flask import Blueprint, jsonify, redirect, render_template, request, session, url_for

from .extensions import db
from .models.experiment import (
    ANNOTATED_APPENDIX_DEFAULT,
    SIMPLIFIED_APPENDIX_DEFAULT,
    VALID_WMN_TYPES,
    Experiment,
    ExperimentDialogue,
    Prompt,
    Run,
    RunResult,
    UserSettings,
)

bp = Blueprint("portal", __name__)

_VALID_WMN_VALUES = {v for v, _ in VALID_WMN_TYPES}


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


@bp.get("/login")
@bp.post("/login")
def login():
    if session.get("user"):
        return redirect(url_for("portal.dashboard"))

    error = None
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        users = _get_users()
        if email in users and users[email] == password:
            session.permanent = True
            session["user"] = email
            return redirect(url_for("portal.dashboard"))
        error = "Invalid email or password."

    return render_template("portal/login.html", error=error)


@bp.post("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("portal.login"))


@bp.get("/dashboard")
@login_required
def dashboard():
    user = session["user"]
    experiments = (
        Experiment.query.filter_by(user_email=user)
        .order_by(Experiment.created_at.desc())
        .all()
    )
    user_settings = db.session.get(UserSettings, user)
    return render_template(
        "portal/dashboard.html",
        user=user,
        experiments=experiments,
        user_settings=user_settings,
        simplified_appendix_default=SIMPLIFIED_APPENDIX_DEFAULT,
        annotated_appendix_default=ANNOTATED_APPENDIX_DEFAULT,
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
    user_settings.annotated_appendix = (request.form.get("annotated_appendix") or "").strip() or None
    db.session.commit()
    return redirect(url_for("portal.dashboard"))


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
            db.session.commit()
            return redirect(url_for("portal.experiment", experiment_id=exp.id))
    return render_template("portal/new_experiment.html", user=user, error=error)


@bp.get("/experiments/<int:experiment_id>")
@login_required
def experiment(experiment_id: int):
    user = session["user"]
    exp = Experiment.query.filter_by(id=experiment_id, user_email=user).first_or_404()
    corpora = _get_corpora()

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

    return render_template(
        "portal/experiment.html",
        user=user,
        experiment=exp,
        corpora=corpora,
        wmn_type_options=VALID_WMN_TYPES,
        latest_runs=latest_runs,
        active_run=active_run,
    )


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

    error = None
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        host = (request.form.get("host") or "").strip()
        model = (request.form.get("model") or "").strip()
        prompt_text = (request.form.get("prompt_text") or "").strip()

        if not name or not host or not model or not prompt_text:
            error = "Name, model, and prompt text are required."
        else:
            next_position = (
                db.session.query(db.func.max(Prompt.position))
                .filter_by(experiment_id=exp.id)
                .scalar()
                or 0
            ) + 1

            system_prompt = (request.form.get("system_prompt") or "").strip() or None

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

            output_format = request.form.get("output_format") or None
            if output_format not in ("simplified", "annotated"):
                output_format = None

            db.session.add(Prompt(
                experiment_id=exp.id,
                position=next_position,
                name=name,
                host=host,
                model=model,
                output_format=output_format,
                system_prompt=system_prompt,
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
        error=error,
    )


@bp.get("/experiments/<int:experiment_id>/prompts/<int:prompt_id>/edit")
@bp.post("/experiments/<int:experiment_id>/prompts/<int:prompt_id>/edit")
@login_required
def edit_prompt(experiment_id: int, prompt_id: int):
    user = session["user"]
    exp = Experiment.query.filter_by(id=experiment_id, user_email=user).first_or_404()
    prompt = Prompt.query.filter_by(id=prompt_id, experiment_id=exp.id).first_or_404()

    error = None
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        host = (request.form.get("host") or "").strip()
        model = (request.form.get("model") or "").strip()
        prompt_text = (request.form.get("prompt_text") or "").strip()

        if not name or not host or not model or not prompt_text:
            error = "Name, model, and prompt text are required."
        else:
            prompt.name = name
            prompt.host = host
            prompt.model = model
            prompt.prompt_text = prompt_text
            prompt.system_prompt = (request.form.get("system_prompt") or "").strip() or None

            output_format = request.form.get("output_format") or None
            if output_format not in ("simplified", "annotated"):
                output_format = None
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
        error=error,
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
    return render_template(
        "portal/results.html",
        user=user,
        experiment=exp,
        prompt=prompt,
        run=run,
        results=results,
        hit_count=hit_count,
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

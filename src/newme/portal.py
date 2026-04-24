from __future__ import annotations

import functools

from flask import Blueprint, redirect, render_template, request, session, url_for

from .extensions import db
from .models.experiment import Experiment

bp = Blueprint("portal", __name__)


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
    return render_template("portal/dashboard.html", user=user, experiments=experiments)


@bp.get("/experiments/<int:experiment_id>")
@login_required
def experiment(experiment_id: int):
    user = session["user"]
    exp = Experiment.query.filter_by(id=experiment_id, user_email=user).first_or_404()
    return render_template("portal/experiment.html", user=user, experiment=exp)

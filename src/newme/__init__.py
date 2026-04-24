from __future__ import annotations

import os
from pathlib import Path

import click
from flask import Flask, Response, redirect, request, url_for
from sqlalchemy import inspect
from sqlalchemy.exc import SQLAlchemyError

from .config import Config
from .extensions import db


def create_app(test_config: dict | None = None) -> Flask:
    _load_environment_from_file()

    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(Config)
    _update_config_from_runtime_env(app)

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    # Optional deploy-local configuration in instance/config.py
    app.config.from_pyfile("config.py", silent=True)

    if test_config is not None:
        app.config.update(test_config)

    _configure_data_path_and_db(app)

    db.init_app(app)

    from .main import bp as main_bp
    from .setup import bp as setup_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(setup_bp)

    @app.before_request
    def enforce_setup_gate() -> Response | None:
        if not app.config.get("REQUIRE_SETUP", True):
            return None

        endpoint = request.endpoint or ""
        if endpoint == "static" or endpoint.startswith("setup."):
            return None

        if not _is_initialized():
            return redirect(url_for("setup.setup_index"))

        return None

    @app.cli.command("install")
    @click.option("--skip-corpora", is_flag=True, help="Skip corpus download and extraction.")
    @click.option("--skip-annotations", is_flag=True, help="Skip annotation import.")
    def install_command(skip_corpora: bool, skip_annotations: bool) -> None:
        """Initialize database and mark the app as installed."""
        from .installation import perform_installation

        result = perform_installation(
            install_corpora=not skip_corpora,
            install_annotations=not skip_annotations,
            logger=click.echo,
        )
        corpora_result = result.get("corpora")
        if isinstance(corpora_result, dict):
            if "dialogue_counts" in corpora_result:
                click.echo(f"Corpora stored: {corpora_result['dialogue_counts']}")
            failed_corpora = corpora_result.get("failed_corpora", {})
            if failed_corpora:
                click.echo(f"Corpora failed: {failed_corpora}")
        if result.get("annotations"):
            click.echo(
                "Annotations stored: "
                f"{result['annotations']['sequence_count']} sequences, "
                f"{result['annotations']['label_count']} labels"
            )
        for warning in result.get("warnings", []):
            click.echo(f"Warning: {warning}")
        click.echo("Installation complete.")

    return app


def _is_initialized() -> bool:
    from .models.app_state import AppState

    try:
        if not inspect(db.engine).has_table(AppState.__tablename__):
            return False
        return AppState.is_initialized()
    except SQLAlchemyError:
        return False


def _configure_data_path_and_db(app: Flask) -> None:
    data_path = (
        app.config.get("DATA_PATH") or os.getenv("NEWME_DATA_PATH") or os.getenv("DATA_PATH")
    )
    if data_path and not app.config.get("DATA_PATH"):
        app.config["DATA_PATH"] = data_path

    database_url = app.config.get("SQLALCHEMY_DATABASE_URI") or os.getenv("DATABASE_URL")
    if database_url and not app.config.get("SQLALCHEMY_DATABASE_URI"):
        app.config["SQLALCHEMY_DATABASE_URI"] = database_url

    require_data_path = app.config.get("REQUIRE_DATA_PATH", True)

    if not data_path and require_data_path:
        raise RuntimeError(
            "DATA_PATH is required. Set NEWME_DATA_PATH (or DATA_PATH) to a writable directory."
        )

    if data_path:
        resolved = Path(data_path).expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        app.config["DATA_PATH"] = str(resolved)
        app.config.setdefault("CORPORA_PATH", str(resolved / "corpora"))
        app.config.setdefault(
            "CORPORA_SOURCE_DIR", str(Path(app.config["CORPORA_PATH"]) / "original_corpora")
        )
        Path(app.config["CORPORA_PATH"]).mkdir(parents=True, exist_ok=True)
        data_path = app.config["DATA_PATH"]

    resolved_annotations_path = _resolve_annotations_path(app.config.get("ANNOTATIONS_PATH"))
    if resolved_annotations_path is not None:
        app.config["ANNOTATIONS_PATH"] = str(resolved_annotations_path)
    else:
        app.config.pop("ANNOTATIONS_PATH", None)

    resolved_corpora_annotations_path = _resolve_annotations_path(
        app.config.get("CORPORA_ANNOTATIONS_PATH")
    )
    if resolved_corpora_annotations_path is not None:
        app.config["CORPORA_ANNOTATIONS_PATH"] = str(resolved_corpora_annotations_path)
    elif app.config.get("ANNOTATIONS_PATH"):
        app.config["CORPORA_ANNOTATIONS_PATH"] = app.config["ANNOTATIONS_PATH"]

    if not app.config.get("SQLALCHEMY_DATABASE_URI"):
        if not data_path:
            raise RuntimeError(
                "SQLALCHEMY_DATABASE_URI is not set and DATA_PATH is unavailable for SQLite fallback."
            )
        app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{Path(data_path) / 'newme.sqlite3'}"


def _load_environment_from_file() -> None:
    env_file = os.getenv("NEWME_ENV_FILE")
    if env_file:
        candidates = [env_file]
    else:
        candidates = [".env", ".env.example"]

    env_path: Path | None = None
    for candidate in candidates:
        candidate_path = Path(candidate).expanduser()
        if not candidate_path.is_absolute():
            candidate_path = (Path.cwd() / candidate_path).resolve()
        if candidate_path.is_file():
            env_path = candidate_path
            break

    if env_path is None:
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[7:].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


def _update_config_from_runtime_env(app: Flask) -> None:
    _set_config_if_present(app, "ENV_FILE", os.getenv("NEWME_ENV_FILE"))
    _set_config_if_present(app, "SECRET_KEY", os.getenv("SECRET_KEY"))
    _set_config_if_present(
        app,
        "ANNOTATIONS_PATH",
        os.getenv("NEWME_ANNOTATIONS_PATH") or os.getenv("ANNOTATIONS_PATH"),
    )
    _set_config_if_present(app, "SQLALCHEMY_DATABASE_URI", os.getenv("DATABASE_URL"))

    runtime_data_path = os.getenv("NEWME_DATA_PATH") or os.getenv("DATA_PATH")
    _set_config_if_present(app, "DATA_PATH", runtime_data_path)

    _set_config_if_present(
        app,
        "INSTALL_CORPORA_ON_SETUP",
        _parse_bool_value(os.getenv("NEWME_INSTALL_CORPORA_ON_SETUP")),
    )
    _set_config_if_present(
        app,
        "INSTALL_ANNOTATIONS_ON_SETUP",
        _parse_bool_value(os.getenv("NEWME_INSTALL_ANNOTATIONS_ON_SETUP")),
    )
    _set_config_if_present(
        app,
        "CORPORA_ENABLED",
        _parse_list_value(os.getenv("NEWME_CORPORA_ENABLED")),
    )
    _set_config_if_present(app, "CORPORA_ANNOTATIONS_PATH", os.getenv("NEWME_CORPORA_ANNOTATIONS_PATH"))
    _set_config_if_present(
        app,
        "CORPORA_DIALOGUE_IDS",
        _parse_list_value(os.getenv("NEWME_CORPORA_DIALOGUE_IDS")),
    )
    _set_config_if_present(app, "CORPORA_CONFIG_PATH", os.getenv("NEWME_CORPORA_CONFIG_PATH"))

    timeout_value = os.getenv("NEWME_CORPORA_TIMEOUT_SECONDS")
    if timeout_value is not None:
        try:
            app.config["CORPORA_TIMEOUT_SECONDS"] = int(timeout_value)
        except ValueError as exc:
            raise RuntimeError("NEWME_CORPORA_TIMEOUT_SECONDS must be an integer.") from exc

    _set_config_if_present(
        app,
        "CORPORA_FORCE_REDOWNLOAD",
        _parse_bool_value(os.getenv("NEWME_CORPORA_FORCE_REDOWNLOAD")),
    )


def _set_config_if_present(app: Flask, key: str, value: object | None) -> None:
    if value is not None:
        app.config[key] = value


def _resolve_annotations_path(configured_path: object | None) -> Path | None:
    for candidate in _annotation_path_candidates(configured_path):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _annotation_path_candidates(configured_path: object | None) -> list[Path]:
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parent.parent
    candidates: list[Path] = []

    configured = str(configured_path).strip() if configured_path is not None else ""
    if configured:
        configured_path_obj = Path(configured).expanduser()
        if configured_path_obj.is_absolute():
            candidates.append(configured_path_obj)
        else:
            candidates.append((Path.cwd() / configured_path_obj).resolve())
            candidates.append((project_root / configured_path_obj).resolve())

    defaults = [
        Path.cwd() / "wmn_annotations.json",
        project_root / "wmn_annotations.json",
        Path.cwd() / "legacy_src" / "newme.old" / "annotation" / "wmn_annotations.json",
        project_root / "legacy_src" / "newme.old" / "annotation" / "wmn_annotations.json",
        package_root / "data" / "wmn_annotations.json",
    ]

    seen: set[str] = set()
    ordered_unique_candidates: list[Path] = []
    for candidate in [*candidates, *defaults]:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        ordered_unique_candidates.append(candidate)
    return ordered_unique_candidates


def _parse_bool_value(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_list_value(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]

from flask import Blueprint, jsonify

from .models.app_state import AppState

bp = Blueprint("setup", __name__, url_prefix="/setup")


@bp.get("/")
def setup_index():
    return jsonify(
        {
            "setup_required": not AppState.is_initialized_safe(),
            "next_step": "Run flask --app newme.wsgi install to initialize database",
        }
    )


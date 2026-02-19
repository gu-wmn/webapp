from flask import Blueprint, jsonify

from .installation import perform_installation
from .models.app_state import AppState

bp = Blueprint("setup", __name__, url_prefix="/setup")


@bp.get("/")
def setup_index():
    return jsonify(
        {
            "setup_required": not AppState.is_initialized_safe(),
            "next_step": "POST /setup to initialize database",
        }
    )


@bp.post("/")
def run_setup():
    result = perform_installation()
    return jsonify(
        {
            "initialized": True,
            "corpora": result.get("corpora"),
            "annotations": result.get("annotations"),
        }
    )

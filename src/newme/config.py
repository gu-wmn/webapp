import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value is None:
        return default
    parsed = [item.strip() for item in value.split(",")]
    return [item for item in parsed if item]


class Config:
    ENV_FILE = os.getenv("NEWME_ENV_FILE", ".env")
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
    DATA_PATH = os.getenv("NEWME_DATA_PATH") or os.getenv("DATA_PATH")
    ANNOTATIONS_PATH = os.getenv("NEWME_ANNOTATIONS_PATH") or os.getenv("ANNOTATIONS_PATH")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    REQUIRE_SETUP = True
    REQUIRE_DATA_PATH = True
    INSTALL_CORPORA_ON_SETUP = _env_bool("NEWME_INSTALL_CORPORA_ON_SETUP", True)
    INSTALL_ANNOTATIONS_ON_SETUP = _env_bool("NEWME_INSTALL_ANNOTATIONS_ON_SETUP", True)
    CORPORA_ENABLED = _env_list(
        "NEWME_CORPORA_ENABLED",
        ["bnc", "winning-args-corpus", "switchboard-corpus"],
    )
    CORPORA_ANNOTATIONS_PATH = os.getenv("NEWME_CORPORA_ANNOTATIONS_PATH")
    CORPORA_DIALOGUE_IDS = _env_list("NEWME_CORPORA_DIALOGUE_IDS", [])
    CORPORA_CONFIG_PATH = os.getenv("NEWME_CORPORA_CONFIG_PATH")
    CORPORA_TIMEOUT_SECONDS = int(os.getenv("NEWME_CORPORA_TIMEOUT_SECONDS", "120"))
    CORPORA_FORCE_REDOWNLOAD = _env_bool("NEWME_CORPORA_FORCE_REDOWNLOAD", False)
    USERS = os.getenv("NEWME_USERS", "")

# newme

New Flask rewrite scaffold with SQL-backed setup gating.

## Project layout

- Active package: `src/newme`
- Legacy code archive: `legacy_src/newme.old`

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export NEWME_DATA_PATH=/absolute/path/to/newme-data
export PYTHONPATH=./src
export FLASK_APP=newme.wsgi
flask run
```

## Installation flow

The app treats installation as database initialization.

- Before installation, regular routes redirect to `/setup/`.
- Complete installation with one of:
  - `flask install`
  - `POST /setup/`
- `flask install` downloads configured corpora and stores extracted data in SQL tables.
- `flask install` also imports `wmn_annotations.json` into SQL tables by default.
- Use `flask install --skip-corpora --skip-annotations` to initialize DB/state only.

After installation, `/` returns the normal application response.
The demo UI is available at:

- `/` browse page with old-style filters and grouped summaries
- `/wmn/<dialogue_id>/<wmn_id>/` WMN sequence page
- `/dialogue/<dialogue_id>` dialogue page
- `/label/<excerpt_hash>` label page

## Configuration

Use environment variables for deployment:

- `NEWME_ENV_FILE` (optional; path to env file loaded at startup)
  If unset, startup checks `./.env` first, then `./.env.example`.
- `NEWME_DATA_PATH` (required; used for SQLite fallback and corpora storage)
- `DATABASE_URL` (optional; if omitted, SQLite is created at `$NEWME_DATA_PATH/newme.sqlite3`)
- `SECRET_KEY`
- `NEWME_INSTALL_CORPORA_ON_SETUP` (default: `true`)
- `NEWME_INSTALL_ANNOTATIONS_ON_SETUP` (default: `true`)
- `NEWME_ANNOTATIONS_PATH` (path to annotation JSON; default auto-detected)
- `NEWME_CORPORA_ENABLED` (comma-separated; default: `bnc,winning-args-corpus,switchboard-corpus`)
- `NEWME_CORPORA_ANNOTATIONS_PATH` (optional; if set, only matching dialogue IDs are extracted)
- `NEWME_CORPORA_DIALOGUE_IDS` (optional comma-separated filter IDs)
- `NEWME_CORPORA_CONFIG_PATH` (optional JSON overrides for corpus URLs/md5/etc)
- `NEWME_CORPORA_TIMEOUT_SECONDS` (default: `120`)
- `NEWME_CORPORA_FORCE_REDOWNLOAD` (default: `false`)

Optional instance config file:

- `instance/config.py`

Example `NEWME_CORPORA_CONFIG_PATH` JSON:

```json
{
  "bnc": {
    "download_url": "https://example.org/bnc.zip",
    "md5sum": "expected-md5"
  }
}
```

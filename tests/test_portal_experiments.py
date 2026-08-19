from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from newme import create_app
from newme.extensions import db
from newme.models import AnnotationSequence, Experiment, ExperimentDialogue, Prompt


class NewExperimentTests(unittest.TestCase):
    def test_new_experiment_resolves_dialogue_sample_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "test.sqlite3"
            app = create_app(
                {
                    "TESTING": True,
                    "SECRET_KEY": "test-secret",
                    "REQUIRE_SETUP": False,
                    "REQUIRE_DATA_PATH": False,
                    "DATA_PATH": temp_dir,
                    "SQLALCHEMY_DATABASE_URI": f"sqlite:///{database_path}",
                }
            )

            with app.app_context():
                db.create_all()
                db.session.add(
                    AnnotationSequence(
                        wmn_id="wmn-1",
                        corpus_codename="switchboard-corpus",
                        dialogue_external_id="dialogue-1",
                        context="Spoken interaction",
                        wmn_type="WMN: non-understanding",
                        wmn_meaning="situated meaning",
                        annotator="tester",
                        comment=None,
                        prediction=None,
                    )
                )
                db.session.commit()

            client = app.test_client()
            with client.session_transaction() as session:
                session["user"] = "tester@example.com"

            response = client.post(
                "/experiments/new",
                data={"name": "Auto resolved experiment"},
                follow_redirects=False,
            )

            self.assertEqual(response.status_code, 302)

            with app.app_context():
                experiment = Experiment.query.filter_by(name="Auto resolved experiment").one()
                prompts = Prompt.query.filter_by(experiment_id=experiment.id).all()
                dialogues = ExperimentDialogue.query.filter_by(experiment_id=experiment.id).all()

                self.assertIsNotNone(experiment.dialogues_resolved_at)
                self.assertEqual(len(prompts), 2)
                self.assertTrue(all(prompt.include_global_template for prompt in prompts))
                self.assertEqual(len(dialogues), 1)
                self.assertEqual(dialogues[0].dialogue_external_id, "dialogue-1")
                self.assertEqual(dialogues[0].corpus_codename, "switchboard-corpus")


if __name__ == "__main__":
    unittest.main()

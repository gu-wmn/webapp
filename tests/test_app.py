import csv
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from html.parser import HTMLParser

from newme import create_app
from newme.extensions import db
from newme.main import _annotate_utterances
from newme.models import AnnotationLabel, AnnotationSequence, Corpus, Dialogue, Utterance


class SpanParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.characters = []
        self.ids = []

    def handle_starttag(self, tag, attrs):
        if tag == 'span':
            attrs = dict(attrs)
            self.stack.append(attrs['class'])
            if 'id' in attrs:
                self.ids.append(attrs['id'])

    def handle_endtag(self, tag):
        if tag == 'span':
            self.stack.pop()

    def handle_data(self, data):
        self.characters.extend((char, set(self.stack)) for char in data)


class AppTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app({'TESTING': True, 'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
                               'REQUIRE_SETUP': False, 'REQUIRE_DATA_PATH': False})
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.create_all()
        corpus = Corpus(codename='test', fullname='Test corpus')
        dialogue = Dialogue(corpus=corpus, external_id='dialogue')
        db.session.add(Utterance(dialogue=dialogue, position=0, author='A', text='<script>alert(1)</script> hello'))
        seq = AnnotationSequence(wmn_id='seq-one', corpus_codename='test', dialogue_external_id='dialogue',
                                 context='Online interaction', wmn_type='WMN: non-understanding',
                                 wmn_meaning='', annotator='A')
        excerpt = 'hello'
        seq.labels.append(AnnotationLabel(position=0, name='Trigger', start_index=0, end_index=0,
                                           start_offset=26, end_offset=31, excerpt=excerpt,
                                           excerpt_hash=hashlib.md5(excerpt.encode()).hexdigest()))
        db.session.add(seq)
        db.session.commit()
        self.client = self.app.test_client()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.ctx.pop()

    def test_sequence_escapes_dialogue_html(self):
        response = self.client.get('/wmn/dialogue/seq-one/')
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b'<script>alert(1)</script>', response.data)
        self.assertIn(b'&lt;script&gt;alert(1)&lt;/script&gt;', response.data)
        self.assertIn(b'id="label-0"', response.data)

    def test_crossing_spans_have_correct_coverage(self):
        labels = [dict(name='Trigger', start_index=0, end_index=0, start_offset=0, end_offset=4, excerpt='abcd'),
                  dict(name='Indicator', start_index=0, end_index=0, start_offset=2, end_offset=6, excerpt='cdef')]
        utterances, links = _annotate_utterances([{'text': 'abcdef'}], labels)
        parser = SpanParser()
        parser.feed(utterances[0]['text'])
        self.assertEqual(parser.characters, [('a', {'Trigger'}), ('b', {'Trigger'}),
                         ('c', {'Trigger', 'Indicator'}), ('d', {'Trigger', 'Indicator'}),
                         ('e', {'Indicator'}), ('f', {'Indicator'})])
        self.assertEqual(parser.stack, [])
        self.assertCountEqual(parser.ids, [link['link'] for link in links])

    def test_filtering_and_csv_export(self):
        self.assertIn(b'seq-one', self.client.get('/?search=hello').data)
        self.assertNotIn(b'seq-one', self.client.get('/?search=absent').data)
        response = self.client.get('/download.csv?search=hello')
        rows = list(csv.reader(io.StringIO(response.get_data(as_text=True))))
        self.assertEqual(len(rows), 2)
        self.assertIn('seq-one', rows[1])
        self.assertEqual(len(list(csv.reader(io.StringIO(self.client.get('/download.csv?search=absent').get_data(as_text=True))))), 1)

    def test_setup_cannot_be_run_through_http(self):
        self.assertEqual(self.client.post('/setup/').status_code, 405)


class InstallationTests(unittest.TestCase):
    def test_fresh_install_stamps_migrations(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app({'TESTING': True, 'DATA_PATH': directory,
                              'SQLALCHEMY_DATABASE_URI': 'sqlite:///' + directory + '/test.sqlite',
                              'MIGRATIONS_PATH': str(Path(__file__).resolve().parents[1] / 'migrations')})
            result = app.test_cli_runner().invoke(args=['install', '--skip-corpora', '--skip-annotations'])
            self.assertEqual(result.exit_code, 0, result.output)
            with app.app_context():
                from sqlalchemy import text
                self.assertIsNotNone(db.session.execute(text('SELECT version_num FROM alembic_version')).scalar())

if __name__ == '__main__':
    unittest.main()

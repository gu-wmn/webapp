import flask
from werkzeug.middleware.proxy_fix import ProxyFix

import newme.annotation as annotation
from newme.corpus.types import CorpusName
from newme.corpus.models import Utterance


class App():

    def __init__(self, data_path: str = "./"):

        self.site_title = "Not Just Semantics"

        self.annotation = annotation.Annotation(data_path=data_path)


    def run(self, test_config=None):

        app = flask.Flask(__name__)

        # Only enable ProxyFix if running behind a proxy (e.g., in production)
        #if os.getenv("USE_PROXY_FIX", "false").lower() == "true":
        app.wsgi_app = ProxyFix(
            app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
        )
        app.debug = True


        @app.before_request
        def fix_path():
            '''
            Fix path if path to app is a subdir determined by a web server.
            Will only do anything if receiving X-Script-Name
            '''
            script_name = flask.request.headers.get('X-Script-Name')
            if script_name:
                flask.request.environ['SCRIPT_NAME'] = script_name


        @app.route("/", methods=['GET'])
        def main_page():

            self.annotation.filter = annotation.models.Filter()

            if flask.request.method == 'GET':
                for param, filter in flask.request.args.items():
                    if any(param == item.name for item in annotation.types.WMNType):
                        self.annotation.filter.wmn_types.append(annotation.types.WMNType[filter])
                    elif param == "search":
                        if filter != "":
                            self.annotation.filter.text_includes = filter
                    elif param == "label-name":
                        if any(filter == item.name for item in annotation.types.LabelName):
                            self.annotation.filter.label_name = annotation.types.LabelName[filter]
                    elif param == "context":
                        if any(filter == item.name for item in annotation.types.Context):
                            self.annotation.filter.context = annotation.types.Context[filter]
                    elif param == "corpus":
                        if any(filter == item.name for item in CorpusName):
                            self.annotation.filter.corpus_codename = CorpusName[filter]
                    elif param == "group-by":
                        if any(filter == item.name for item in annotation.types.GroupBy):
                            self.annotation.filter.group_by = annotation.types.GroupBy[filter]
                    else:
                        print("Unkown parameter:", param, filter, flush=True)

            summaries = self.annotation.get_summaries()

            label_names = [
                annotation.types.LabelName.TRIGGER,
                annotation.types.LabelName.INDICATOR
            ]

            return flask.render_template(
                'index.jinja',
                site_title = self.site_title,
                LabelName = annotation.types.LabelName,
                wmn_options = annotation.types.WMNType,
                Corpus = CorpusName,
                labels = label_names,
                context_labels = annotation.types.Context,
                groupings = [ # specific order
                    annotation.types.GroupBy.DIALOGUE,
                    annotation.types.GroupBy.SEQUENCE,
                    annotation.types.GroupBy.LABEL
                ],
                GroupBy = annotation.types.GroupBy,
                num_results = len(summaries),
                summaries = summaries,
                get_data = flask.request.args
            )


        @app.route("/dialogue/<dialogue_id>")
        def dialogue_page(dialogue_id: str):
            # get meta information about the corpus
            # and all wmns attached to the corpus
            return flask.render_template(
                'dialogue/index.jinja',
                site_title = self.site_title,
                dialogue_metadata = self.annotation.get_dialogue_metadata(dialogue_id)
            )


        @app.route("/wmn/<dialogue_id>/<wmn_id>/")
        def sequence_page(
            dialogue_id: str,
            wmn_id: str
        ):

            wmn_sequence = self.annotation.get_wmn_sequence(dialogue_id, wmn_id)
            wmn_sequence.labels.sort(
                key=lambda label: (
                    label['start_index'],
                    label['start_offset']
                )
            )
            wmn_sequence.utterances, label_links = annotate_utterances(
                wmn_sequence.utterances,
                wmn_sequence.labels
            )

            return flask.render_template(
                'wmn_sequence.jinja',
                site_title = self.site_title,
                wmn_sequence = wmn_sequence,
                labels = wmn_sequence.labels,
                label_links = label_links
            )


        @app.route("/label/<excerpt>")
        def label_page(excerpt):
            label_metadata = self.annotation.get_label_metadata(excerpt=excerpt)
            print(label_metadata, flush=True)
            return flask.render_template(
                'label.jinja',
                label = label_metadata
            )


        @app.route("/about")
        def about_page():
            return flask.render_template(
                'about.jinja',
                site_title=self.site_title
            )

        return app


def annotate_utterances(
    utterances: list[type[Utterance]],
    labels: list[dict]
):
    label_links = []

    spans = {
        "Trigger": ['<span class="Trigger" id="label-', '">'],
        "Indicator": ['<span class="Indicator" id="label-', '">'],
        "Negotiation": ['<span class="Negotiation" id="label-', '">'],
        "end": '</span>'
    }

    inserts = []

    label_id = 0

    for label in labels:

        start_index = int(label['start_index'])
        end_index = int(label['end_index'])

        add_to_start = 0
        # adjust start
        for insert in inserts:
            # check if inserts happened before start_offset in start_index and adjust values
            if (
                insert[0] == start_index
                and insert[1] < int(label['start_offset'])
            ):
                add_to_start += insert[2]

        start_offset = int(label['start_offset']) + add_to_start
        # add start
        span = spans[label['name']][0] + str(label_id) + spans[label['name']][1]

        utterances[start_index].text = \
            utterances[start_index].text[:start_offset] \
            + span \
            + utterances[start_index].text[start_offset:]
        inserts.append((
            start_index,
            int(label['start_offset']),
            len(span)
        ))
        label_links.append({
            'name': label['name'],
            'excerpt': label['excerpt'],
            'link': 'label-' + str(label_id)
        })
        label_id += 1

        # add spans for negotiations that span more then one utterance
        if start_index < end_index:
            utterances[start_index].text = utterances[start_index].text + spans['end']
        if end_index - start_index == 1:
            utterances[start_index + 1].text = span + utterances[start_index].text
            inserts.append((
                start_index + 1,
                0,
                len(span)
            ))
        elif end_index - start_index > 1:
            print("negotiation spanned more than two utterances", flush=True)
            for i in range(start_index + 1, end_index):
                utterances[i].text = span + utterances[i].text + spans['end']
                inserts.append((
                    i,
                    0,
                    len(span)
                ))
                inserts.append((
                    i,
                    len(utterances[i].text),
                    len(span)
                ))
            utterances[end_index].text = span + utterances[end_index].text
            inserts.append((
                end_index,
                0,
                len(span)
            ))

        add_to_end = 0
        # adjust end
        for insert in inserts:
            if (
                insert[0] == end_index
                and insert[1] < int(label['end_offset'])
            ):
                add_to_end += insert[2]

        end_offset = int(label['end_offset']) + add_to_end
        # add end
        utterances[end_index].text = \
            utterances[end_index].text[:end_offset] \
            + spans['end'] \
            + utterances[end_index].text[end_offset:]
        inserts.append((
            label['end_index'],
            label['end_offset'],
            len(spans['end'])
        ))

    return utterances, label_links


def create_app():
    app = App()
    return app.run()

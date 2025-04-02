import flask
from werkzeug.middleware.proxy_fix import ProxyFix
import os
from .annotation import (
    Annotation,
    SearchParameters,
    LabelName,
    WMN,
    Context
)


class App():

    def __init__(self, data_path: str = "./"):

        self.site_title = "Not Just Semantics"

        self.annotation = Annotation(data_path=data_path)


    def run(self, test_config=None):

        app = flask.Flask(
            __name__,
            # template_folder='./templates',
            # static_url_path='/static'
        )
        # Only enable ProxyFix if running behind a proxy (e.g., in production)
        #if os.getenv("USE_PROXY_FIX", "false").lower() == "true":
        app.wsgi_app = ProxyFix(
            app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
        )
        #app.debug = True


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

            search_parameters = SearchParameters(
                wmn = []
            )

            if flask.request.method == 'GET':
                for param, filter in flask.request.args.items():
                    if any(param == item.name for item in WMN):
                        print(param, filter, flush=True)
                        search_parameters.wmn.append(WMN[filter])
                    elif param == "search":
                        if filter != "":
                            search_parameters.text_includes = filter
                    elif param == "label-name":
                        if any(filter == item.name for item in LabelName):
                            search_parameters.label_name = LabelName[filter]
                    elif param == "context":
                        if any(filter == item.name for item in Context):
                            search_parameters.context = Context[filter]

            summaries = self.annotation.match_and_get_summaries(
                search_parameters
            )
            summaries.sort(key=lambda x: x.annotated_utterance.labeled_text.casefold())

            print(flask.request.args, flush=True)
            return flask.render_template(
                'index.html',
                site_title = self.site_title,
                summaries = summaries,
                WMN = WMN,
                LabelName = LabelName,
                Context = Context,
                num_results = len(summaries),
                get_data = flask.request.args
            )


        @app.route("/wmn-sequence/<wmn_id>/")
        def sequence_page(wmn_id: int):

            wmn_sequence = self.annotation.get_wmn_sequence(wmn_id)
            triggers = [label for label in wmn_sequence.labels if LabelName(label["name"]) == LabelName.Trigger]
            indicators = [label for label in wmn_sequence.labels if LabelName(label["name"]) == LabelName.Indicator]
            negotations = [label for label in wmn_sequence.labels if LabelName(label["name"]) == LabelName.Negotiation]
            #trigger_refs = [label for label in wmn_sequence.labels if LabelName(label["name"]) == LabelName.Trigger_Reference]
            labels = triggers + indicators + negotations #+ trigger_refs

            return flask.render_template(
                'wmn_sequence.html',
                site_title = self.site_title,
                wmn_sequence = wmn_sequence,
                labels = labels
            )


        @app.route("/about")
        def about_page():
            return flask.render_template(
                'about.html',
                site_title=self.site_title
            )


        return app


def create_app():
    app = App()
    return app.run()

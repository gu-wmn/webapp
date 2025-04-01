# README

## About this web application

This is a web application for the research project [Not Just Semantics](https://dev.clasp.gu.se/word-negotiation/about).
It applies and displays standoff annotation with a series of corpora.
The standoff annotation can be found at [./src/not_just_semantics/wmn_annotation.json](https://github.com/gu-wmn/webapp/src/not_just_semantics/wmn_annotation.json),
while the corpora will be downloaded at first run and a portion of the corpus text will be extracted into a local json file.

### Corpora

Corpora currently used:

* [British National Corpus](http://www.natcorp.ox.ac.uk/) - [License](http://www.natcorp.ox.ac.uk/docs/licence.html)
* [Winning Arguments (ChangeMyView) Corpus](https://convokit.cornell.edu/documentation/winning.html) - (License Unknown)
* [Switchboard Dialog Act Corpus](http://compprag.christopherpotts.net/swda.html) - [License: CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/)

## Running this web app

Currently the size of the corpora will take up about 6GB of space.

First:
```
git clone <url>
cd word_negotiation/
```

### Run locally with flask in a python venv

When using this method, corpora will be downloaded to current dir.

Do:
```(bash)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
flask --app src/not_just_semantics/app run
```
Then, go to: [http://localhost:5000](http://localhost:5000)


### Running the web application as a docker

Corpora will be downloaded to /srv on the docker. In this example, /srv is mounted to
./ (current dir).

To run the web application with docker, do:
```(bash)
docker build -t not_just_semantics:latest .
docker run -d -v./:/srv -p 127.0.0.1:8000:8000  not_just_semantics:latest
```
Then, go to: [http://localhost:8000](http://localhost:8000)

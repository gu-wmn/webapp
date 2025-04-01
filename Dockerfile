FROM python:3.12

WORKDIR /srv

# Install the word_negotiation source as a pip package
COPY ./src ./src
COPY ./MANIFEST.in .
COPY ./setup.py .

#RUN apt update
#RUN pip install -r requirements.txt
RUN pip install .

# remove the source from /srv
RUN rm -rf ./*

# Start the server
ENV USE_PROXY_FIX = true
CMD ["gunicorn", "--bind=0.0.0.0", "not_just_semantics.app:create_app()"]

FROM python:3.12

WORKDIR /srv

# Install the package
COPY ./src ./src
COPY ./MANIFEST.in .
COPY ./setup.py .
COPY ./requirements.txt .
COPY ./migrations /opt/newme/migrations

RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt
RUN pip install .

# remove the source from /srv
RUN rm -rf ./*

# Start the server
ENV NEWME_MIGRATIONS_PATH=/opt/newme/migrations
CMD ["gunicorn", "--bind=0.0.0.0:8000", "newme.wsgi:app"]

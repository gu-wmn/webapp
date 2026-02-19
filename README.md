# newme

To start container:
```bash
docker compose up
```

To install corpora and extract dialogue parts:
```
docker compose exec web python -m flask --app newme.wsgi install
```

Visit [http://127.0.0.1:8000](http://127.0.0.1:8000)

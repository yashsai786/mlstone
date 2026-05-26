.PHONY: install api worker test test-cov clean docker-build

VENV = ./venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
pytest = $(VENV)/bin/pytest

install:
	$(PIP) install -r requirements.txt

api:
	API_HOST=127.0.0.1 API_PORT=8000 ZMQ_BROKER_URL=tcp://127.0.0.1:5555 $(PYTHON) -m uvicorn src.app.api.main:app --reload

worker:
	ZMQ_BROKER_URL=tcp://127.0.0.1:5555 $(PYTHON) -m src.app.messaging.worker

test:
	$(pytest) -v

test-cov:
	$(pytest) --cov=src -v

clean:
	rm -rf .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

docker-build:
	docker build -t stone-slab-api:latest .

docker-run-api:
	docker run -d --name stone-api -p 8000:8000 -v $$(pwd)/storage:/app/storage stone-slab-api:latest

docker-run-worker:
	docker run -d --name stone-worker --net=host -v $$(pwd)/storage:/app/storage stone-slab-api:latest python -m src.app.messaging.worker

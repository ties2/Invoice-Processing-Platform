.PHONY: install test run up down

install:
	pip install -r requirements.txt

test:
	pytest

run:
	uvicorn src.serving.app:app --reload

up:
	docker compose up --build

down:
	docker compose down

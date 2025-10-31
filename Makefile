.PHONY: dev train drift serve-drift stack-up stack-down

dev:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

train:
	python src/train.py

drift:
	python monitoring/generate_drift.py

serve-drift:
	uvicorn monitoring.evidently_app:app --host 0.0.0.0 --port 7000

stack-up:
	docker compose up -d

stack-down:
	docker compose down

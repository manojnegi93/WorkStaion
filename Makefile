.PHONY: install dev api ui test lint eval eval-judge demo docker-up docker-down

install:            ## install backend + frontend dependencies
	cd backend && pip install -r requirements-dev.txt
	cd frontend && npm install

api:                ## run the FastAPI backend with reload
	cd backend && uvicorn app.main:app --reload --port 8000

ui:                 ## run the React dev server
	cd frontend && npm run dev

dev:                ## run backend and frontend together
	$(MAKE) -j2 api ui

demo:               ## run without any model downloaded (offline extractive mode)
	cd backend && LLM_PROVIDER=offline QDRANT_PATH=:memory: uvicorn app.main:app --port 8000 & cd frontend && npm run dev

test:               ## unit + API tests (offline, deterministic)
	cd backend && python -m pytest -q

lint:
	cd backend && ruff check . && ruff format --check .
	cd frontend && npm run typecheck

eval:               ## evaluation suite + release gate (offline model)
	cd backend && LLM_PROVIDER=offline QDRANT_PATH=:memory: python -m scripts.run_eval

eval-judge:         ## evaluate the real Ollama model with LLM-as-judge
	cd backend && python -m scripts.run_eval --reindex --judge --thresholds data/eval/thresholds.llm.yaml

docker-up:
	cp -n .env.example .env || true
	docker compose up -d --build

docker-down:
	docker compose down

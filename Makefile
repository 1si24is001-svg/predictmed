export PYTHONPATH := src

.PHONY: install pipeline test api dashboard sweep clean

install:
	pip install -r requirements.txt

pipeline:            ## full run: ingest -> ... -> uci  (~15 min on 2 CPUs)
	python -m predictmed.pipeline run

test:
	pytest -q tests

api:
	uvicorn predictmed.serving.api:app --port 8000

frontend-build:
	cd frontend && npm install && npm run build

dashboard: frontend-build
	uvicorn predictmed.serving.api:app --port 8000

sweep:               ## local equivalent of slurm/sweep_array.sbatch
	for m in lightgbm catboost logistic; do python -m predictmed.sweep --model $$m --seed 0; done

clean:
	rm -rf data/lakehouse data/*.parquet data/*.db artifacts reports/* mlruns

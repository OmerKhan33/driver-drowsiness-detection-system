# Driver Drowsiness Detection — pipeline automation
#
# Targets are grouped from "manual ML chores" → "fully automated pipeline".
# Use `make help` for the menu. Variables (PY, PORT, IMAGE) are overridable:
#   make train PY=./venv/Scripts/python.exe
#   make app   PORT=8502

PY     ?= python
PIP    ?= $(PY) -m pip
IMAGE  ?= drowsiness-detector
PORT   ?= 8501

.PHONY: help install data validate manifest train sanity test lint format \
        docker-build docker-run app pipeline clean

help:
	@echo "Common targets:"
	@echo "  make install         install Python deps into the active env"
	@echo "  make data            prepare dataset from data/raw -> data/processed"
	@echo "  make validate        validate processed dataset (CI-gateable)"
	@echo "  make manifest        write models/results/data_manifest.json"
	@echo "  make train           train all CNN architectures"
	@echo "  make sanity          run model sanity checks"
	@echo "  make test            run pytest suite"
	@echo "  make lint            flake8 over src/ app/ data/scripts/"
	@echo "  make format          black + isort"
	@echo "  make docker-build    build the production image"
	@echo "  make docker-run      run the image, exposing $(PORT)"
	@echo "  make app             run streamlit locally on $(PORT)"
	@echo "  make pipeline        full pipeline: data -> validate -> manifest -> train -> sanity -> test"
	@echo "  make clean           remove caches and __pycache__"

install:
	$(PIP) install -r requirements.txt

data:
	$(PY) data/scripts/prepare_dataset.py

validate:
	$(PY) data/scripts/validate_dataset.py --report models/results/validation_report.json

manifest:
	$(PY) data/scripts/build_manifest.py --output models/results/data_manifest.json

train:
	$(PY) src/classification/train.py --epochs 15 --batch_size 32 --lr 0.0001

sanity:
	$(PY) src/utils/sanity_check.py

test:
	$(PY) -m pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	$(PY) -m flake8 src/ app/ data/scripts/ --max-line-length=120 --ignore=E203,W503,E501,E402

format:
	$(PY) -m black --line-length=120 src/ app/ tests/ data/scripts/
	$(PY) -m isort --profile black src/ app/ tests/ data/scripts/

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p $(PORT):8501 -v $(CURDIR)/data:/app/data $(IMAGE)

app:
	$(PY) -m streamlit run app/streamlit_app.py \
	    --server.address localhost --server.port $(PORT) \
	    --server.headless true --browser.gatherUsageStats false

pipeline: data validate manifest train sanity test
	@echo "Full pipeline complete."

clean:
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	@find . -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +
	@rm -f coverage.xml .coverage
PYTHON ?= .venv/bin/python
BT = .venv/bin/btl11
CONFIG ?= configs/base.yaml

.PHONY: install test lint smoke audit prepare eda pca baseline ablations stability verify verify-all controlled-gmm robustness report-help

install:
	test -x $(PYTHON) || python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest -q

lint:
	.venv/bin/ruff check src tests scripts

smoke:
	$(BT) --help
	$(PYTHON) -m btl11 --help

audit:
	$(BT) audit --input data/raw/dataset.csv --config $(CONFIG)

prepare:
	$(BT) prepare --config $(CONFIG)

eda:
	$(BT) eda --config $(CONFIG)

pca:
	$(BT) pca --config $(CONFIG)

baseline:
	$(BT) sweep --algorithm kmeans --config $(CONFIG)
	$(BT) sweep --algorithm gmm --config $(CONFIG)
	$(BT) compare --experiment-set baseline_f9_standard --config $(CONFIG)

ablations:
	$(BT) ablate --factor scaler --config $(CONFIG)
	$(BT) ablate --factor feature_set --config $(CONFIG)
	$(BT) ablate --factor pca --config $(CONFIG)

stability:
	$(BT) stability --model-selector finalists --config $(CONFIG)

verify:
	@test -n "$(RUN_ID)" || (echo "Usage: make verify RUN_ID=<run-id>" && exit 2)
	$(BT) verify --run-id $(RUN_ID) --config $(CONFIG)

verify-all:
	@for rid in $$($(PYTHON) -c "import pandas as pd; print(' '.join(pd.read_csv('artifacts/INDEX.csv')['run_id'].astype(str)))"); do \
		$(BT) verify --run-id $$rid --config $(CONFIG) >/dev/null || exit 1; \
	done
	@echo "All runs in artifacts/INDEX.csv verified."

controlled-gmm:
	$(PYTHON) scripts/run_controlled_gmm.py --config $(CONFIG)

robustness:
	$(BT) robustness --config $(CONFIG)

report-help:
	@echo "Run: $(BT) report --selected-run <FINAL_REFIT_RUN_ID> --output artifacts/final-report-tables --config $(CONFIG)"

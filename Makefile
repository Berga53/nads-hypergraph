PYTHON ?= python3
VENV_PYTHON := .venv/bin/python

.PHONY: setup prepare-data run-experiment run-geographical run-starting-point test

setup:
	$(PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt

prepare-data:
	$(VENV_PYTHON) src/prepare_data.py

run-experiment:
	$(VENV_PYTHON) scripts/run_experiment.py

run-geographical:
	$(VENV_PYTHON) scripts/run_geographical_experiment.py

run-starting-point:
	$(VENV_PYTHON) scripts/run_starting_point_experiment.py

test:
	MPLCONFIGDIR=/tmp/rete-ipl-mpl MPLBACKEND=Agg $(VENV_PYTHON) -m unittest discover -s tests -v

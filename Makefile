PY ?= python3

.PHONY: help test lint experiment calibrate measure policies report clean

help:
	@echo "make test        run the test suite"
	@echo "make experiment  calibrate, measure every workload, replay, summarise"
	@echo "make report      rebuild results/summary.md from existing JSON"
	@echo "make clean       remove caches (keeps results/)"

WORKLOADS = one-shot validate log-lines bulk adversarial zipf \
            mix-0 mix-50 mix-200 mix-800 mix-2000 mix-5000

test:
	$(PY) -m pytest tests -q

lint:
	$(PY) -m pyflakes hp tests tools || true

calibrate:
	$(PY) -m hp.eval.calibrate

measure: calibrate
	@for w in $(WORKLOADS); do $(PY) -m hp.eval.run measure $$w; done

policies:
	@for w in $(WORKLOADS); do echo "== $$w"; $(PY) -m hp.eval.run policies $$w; done

report:
	$(PY) tools/summarise.py

experiment: measure policies report
	@echo
	@echo "results/summary.md rebuilt."

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache build dist *.egg-info

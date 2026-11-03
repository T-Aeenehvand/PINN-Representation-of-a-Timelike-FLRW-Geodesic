PYTHON ?= python

.PHONY: test verify smoke figures upgraded-figures extension-figures

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

verify:
	$(PYTHON) verify_release.py

smoke:
	FLRW_PROFILE=smoke FLRW_RESULTS_ROOT=results/local_runs PYTHONPATH=src $(PYTHON) scripts/run_main_ablation.py

figures:
	PYTHONPATH=src $(PYTHON) scripts/regenerate_figures.py

upgraded-figures:
	PYTHONPATH=src $(PYTHON) scripts/regenerate_upgraded_figures.py

extension-figures:
	$(PYTHON) scripts/plot_dimensional_extension.py --results results/dimensional_extension --out figures/dimensional_extension

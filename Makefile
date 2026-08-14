.PHONY: exposure test smoke

exposure:
	python -m src.build_exposure
	python analysis/01_exposure_descriptives.py

test:
	python -m unittest discover -s tests -v

smoke: exposure
	python tools/make_synthetic_cps.py
	python -m src.build_panel --baseline 2022 --recent 2025
	python analysis/02_young_workers.py --allow-synthetic
	python -m src.elsi --allow-synthetic
	python analysis/04_regressions.py --post-from 2024 --allow-synthetic

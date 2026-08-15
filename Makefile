.PHONY: exposure flows research test smoke

exposure:
	python -m src.build_exposure
	python analysis/01_exposure_descriptives.py

test:
	python -m unittest discover -s tests -v

flows: exposure
	python analysis/03_worker_flows.py --post-from 2024

research: flows
	python -m src.build_panel --baseline 2022 --recent 2025
	python analysis/02_young_workers.py
	python analysis/03_headline.py
	python analysis/04_regressions.py --post-from 2024 --base-year 2023
	python analysis/05_robustness.py --post-from 2024
	python -m src.elsi

smoke: exposure
	python tools/make_synthetic_cps.py
	python -m src.build_panel --baseline 2022 --recent 2025
	python analysis/02_young_workers.py --allow-synthetic
	python analysis/03_headline.py --allow-synthetic
	python -m src.elsi --allow-synthetic
	python analysis/04_regressions.py --post-from 2024 --allow-synthetic
	python analysis/05_robustness.py --post-from 2024 --allow-synthetic

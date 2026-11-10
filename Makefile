.PHONY: install all smoke test docker clean

install:
	pip install -e ".[torch]"

all:
	python scripts/run_all.py --all

smoke:
	python scripts/run_all.py --smoke

test:
	python -m pytest -q

docker:
	docker compose up --build

clean:
	rm -rf results/*.json results/figures/*.png

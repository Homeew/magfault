SHELL := /bin/bash

# Override on the command line, e.g.  make download STATIONS=VIC,MEA START=2024-04-01 END=2024-06-30
STATIONS ?= VIC,MEA,BRD,OTT
START    ?= 2024-01-01
END      ?= 2024-12-31
SEED     ?= 42
RUN      ?=
RAW      ?= data/raw
SYNTH    ?= data/synth

RUN_ARG := $(if $(RUN),--run $(RUN),)

.PHONY: help up down initdb download kp load inject features detect evaluate pipeline demo test lint clean-db

help:
	@echo "Real data:   make up initdb download kp load pipeline"
	@echo "Offline:     make up initdb demo        (synthetic network, no downloads)"
	@echo "Tests:       make test"

up:
	docker compose up -d db
	@until docker compose exec -T db pg_isready -U magfault -d magfault >/dev/null 2>&1; do sleep 1; done
	@echo "Postgres is ready."

down:
	docker compose down

initdb:
	magfault initdb

download:
	magfault download --stations $(STATIONS) --start $(START) --end $(END) --out $(RAW)

kp:
	magfault kp --start $(START) --end $(END) --out $(RAW)/kp

load:
	magfault load $(RAW)

inject:
	magfault inject --seed $(SEED)

features:
	magfault features $(RUN_ARG)

detect:
	magfault detect $(RUN_ARG)

evaluate:
	magfault evaluate $(RUN_ARG)

# Fault injection -> SQL features -> detectors -> evaluation on whatever is loaded.
pipeline: inject features detect evaluate

# End-to-end on a synthetic 4-station network (used by CI too).
demo:
	magfault synth --days 90 --out $(SYNTH)
	magfault load $(SYNTH)
	$(MAKE) pipeline

test:
	pytest

lint:
	ruff check src tests

clean-db:
	magfault initdb --drop

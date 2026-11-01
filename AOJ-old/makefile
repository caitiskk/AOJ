CONFIG ?= configs/default.yaml

.PHONY: all smoke preprocess train evaluate interpret clean

all: train evaluate

smoke:
	python scripts/train.py --config configs/default.yaml

preprocess:
	python scripts/preprocess.py --config $(CONFIG) --force

train:
	python scripts/train.py --config $(CONFIG)

evaluate:
	python scripts/evaluate.py --config $(CONFIG)

interpret:
	python scripts/interpret.py --config $(CONFIG)

clean:
	rm -rf outputs/experiments/smoke_test
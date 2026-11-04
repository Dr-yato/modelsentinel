.PHONY: build test dashboard demo serve

build:
	cargo build --manifest-path native/Cargo.toml
	pip install -e ".[dev]"

test: build
	cargo test --manifest-path native/Cargo.toml
	pytest

dashboard:
	npm --prefix dashboard ci
	npm --prefix dashboard run build

demo: build
	python -m modelsentinel demo

serve: dashboard
	python -m modelsentinel serve --demo --host 127.0.0.1 --port 8741

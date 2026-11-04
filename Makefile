# GNU make. On Windows run the same cargo / npm / pip commands shown in README.
.PHONY: all native dashboard install dev test demo serve doctor clean

PYTHON ?= python
CARGO  ?= cargo
NPM    ?= npm
EXE    := $(if $(filter Windows_NT,$(OS)),.exe,)
BIN    := src/modelsentinel/bin
WEB    := src/modelsentinel/web

all: native dashboard dev

## Build the Rust parser + sandbox (release) and copy them into the package.
native:
	$(CARGO) build --release --manifest-path native/Cargo.toml
	$(PYTHON) -c "import pathlib,shutil; d=pathlib.Path('$(BIN)'); d.mkdir(exist_ok=True); [shutil.copy2(f'native/target/release/{n}$(EXE)', d) for n in ('sentinel-parse','sentinel-sandbox')]"

## Build the dashboard and copy it into the package so `sentinel serve` finds it after install.
dashboard:
	$(NPM) --prefix dashboard ci
	$(NPM) --prefix dashboard run build
	$(PYTHON) -c "import shutil; shutil.rmtree('$(WEB)', ignore_errors=True); shutil.copytree('dashboard/dist', '$(WEB)')"

## Editable install with dev extras (tests).
dev:
	$(PYTHON) -m pip install -e ".[dev]"

## Regular install: setuptools-rust compiles the binaries when cargo is present.
install: dashboard
	$(PYTHON) -m pip install ".[scanners]"

test: native
	$(CARGO) test --manifest-path native/Cargo.toml
	$(PYTHON) -m pytest

demo:
	$(PYTHON) -m modelsentinel demo

serve:
	$(PYTHON) -m modelsentinel serve --demo --host 127.0.0.1 --port 8741

doctor:
	$(PYTHON) -m modelsentinel doctor

clean:
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('$(BIN)', '$(WEB)', 'build', 'dist')]"

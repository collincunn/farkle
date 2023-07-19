clean:
	find . -name '*egg-info' | xargs rm -rf
	find . -name '.coverage' | xargs rm -rf
	find . -name '.mypy_cache' | xargs rm -rf
	find . -name '.pytest_cache' | xargs rm -rf
	find . -name '.tox' | xargs rm -rf
	find . -name '__pycache__' | xargs rm -rf
	find . -name 'reports' | xargs rm -rf
	find . -name '.ruff_cache' | xargs rm -rf
	find . -name '.ipynb_checkpoints' | xargs rm -rf
	find . -name '*.pyc' -delete 2>&1
	find . -name '*.pyo' -delete 2>&1

prerequisites:
	pip3 install -U pip setuptools wheel setuptools_scm[toml]

install:prerequisites
	pip3 install -U --upgrade-strategy eager .

develop:prerequisites
	pip3 install -U --upgrade-strategy eager tox
	pip3 install -U --upgrade-strategy eager -e '.[dev]'

lint:
	pre-commit run --all-files --hook-stage manual

package:prerequisites
	pip3 install -U --upgrade-strategy eager build
	pyproject-build --no-isolation

test:
	tox --recreate --parallel auto

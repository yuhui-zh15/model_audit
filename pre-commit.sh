#!/bin/bash

set -e

pip install -r requirements.txt
pre-commit install

isort src
black src
mypy src
flake8 src

echo "Done."

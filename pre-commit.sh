#!/bin/bash

set -e

pip install -r requirements.txt
pre-commit install

black src
mypy src
flake8 src

echo "Done."

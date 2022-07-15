#!/bin/bash

set -e

pip install black mypy flake8 click pytest pre-commit typed-ast

black src
mypy src
flake8 src
pytest

echo "Done."

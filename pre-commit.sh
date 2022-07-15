#!/bin/bash

set -e

pip install black mypy flake8

black src
mypy src
flake8 src

echo "Done."

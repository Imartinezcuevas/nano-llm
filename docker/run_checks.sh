#!/bin/bash
set -e

echo "Running Ruff..."
ruff check .

echo "Running MyPy..."
mypy .

echo "Running PyTest..."
pytest -q

echo "All checks passed!"

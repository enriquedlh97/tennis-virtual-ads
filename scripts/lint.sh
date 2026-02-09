#!/usr/bin/env bash

set -e
set -x

mypy src scripts
ruff check src scripts
ruff format src scripts --check

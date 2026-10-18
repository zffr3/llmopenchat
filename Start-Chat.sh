#!/bin/sh
set -eu
cd -- "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ ! -x .venv/bin/python ]; then
    printf '%s\n' 'Please run ./Install.sh first.' >&2
    exit 1
fi
exec .venv/bin/python -m llmopenchat "$@"

#!/bin/sh
set -eu
cd -- "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ ! -x .venv/bin/python ]; then
    printf '%s\n' 'Please run ./Install.sh first.' >&2
    exit 1
fi
if [ "${1:-}" = "--config" ]; then
    if [ "$#" -lt 2 ]; then
        printf '%s\n' '--config requires a path.' >&2
        exit 2
    fi
    config_path=$2
    shift 2
    exec .venv/bin/python -m llmopenchat --config "$config_path" telegram "$@"
fi
exec .venv/bin/python -m llmopenchat telegram "$@"

#!/bin/sh
# Set up the client and a managed local model. No root privileges required.
set -eu
cd -- "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
chmod +x Install.sh Start-Chat.sh Start-Telegram.sh

if [ ! -x .venv/bin/python ]; then
    "${PYTHON:-python3}" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt

if [ "${1:-}" = "--client-only" ]; then
    shift
    if [ "$#" -ne 0 ]; then
        printf '%s\n' '--client-only does not accept model installation options.' >&2
        exit 2
    fi
    printf '%s\n' 'Client installed. Configure your external API in config.json.'
else
    .venv/bin/python -m llmopenchat install "$@"
    printf '%s\n' 'Installation complete. Run ./Start-Chat.sh to chat.'
fi

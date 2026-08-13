#!/bin/sh
# Deploy: pull the latest, sync the venv, and restart the service.
#
# `uv sync` runs here (in the user's environment, which has git auth) rather
# than in the unit, because the systemd environment cannot fetch git deps.
set -eu

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

uv=$(command -v uv 2>/dev/null || echo "$HOME/.local/bin/uv")

git pull --ff-only
"$uv" sync --locked
systemctl --user restart financial-dashboard.service
systemctl --user --no-pager status financial-dashboard.service | head -n 8

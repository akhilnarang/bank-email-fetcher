#!/bin/sh
# Deploy: pull the latest and restart the service. `uv run` on restart syncs
# the venv to the pulled uv.lock, so nothing else is needed.
set -eu

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

git pull --ff-only
systemctl --user restart financial-dashboard.service
systemctl --user --no-pager status financial-dashboard.service | head -n 8

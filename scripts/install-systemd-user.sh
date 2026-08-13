#!/bin/sh
# Install the financial-dashboard systemd user unit (one-time per host).
#
# Copies deploy/systemd/financial-dashboard.service into the user unit dir,
# enables it, and starts it. Re-run to pick up unit changes. Stop any manual
# `uv run fastapi run` first so the unit can bind port 8000.
set -eu

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
unit_name=financial-dashboard.service
unit_src=$root/deploy/systemd/$unit_name
unit_dir=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user

install -d -m 0755 "$unit_dir"
install -m 0644 "$unit_src" "$unit_dir/$unit_name"

systemctl --user daemon-reload
systemctl --user enable --now "$unit_name"
systemctl --user --no-pager status "$unit_name" | head -n 8

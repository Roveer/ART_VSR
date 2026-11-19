#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python inference.py --chunk_size 17 --overlap 5 "$@"

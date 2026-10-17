#!/bin/sh
# First start of an empty volume only: the base as the API's start builds it — the history parsed with the
# knowledge, the vectors of the names, the demo verdicts. The vectors need the key to compute, so without one
# they come from here; the API's start then re-parses the rows and keeps them.
set -e
if [ -f /dump/analog.dump ]; then
    pg_restore --no-owner --no-privileges -U "$POSTGRES_USER" -d "$POSTGRES_DB" /dump/analog.dump
fi

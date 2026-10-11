#!/usr/bin/env bash
# Download the real Google Borg 2011 trace shards used by the environment.
# Zero authentication; files are ~4 MB each. Shards are sequential in time.
set -euo pipefail
cd "$(dirname "$0")/.."

PARTS="${1:-0-3}"          # e.g. "0-3" or "0,1,2,5"
DATA_DIR="${DATA_DIR:-data/raw/google2011}"
BASE="https://storage.googleapis.com/clusterdata-2011-2"

mkdir -p "$DATA_DIR/machine_events" "$DATA_DIR/task_events"

echo "==> machine_events"
curl -L --fail --retry 3 -o "$DATA_DIR/machine_events/part-00000-of-00001.csv.gz" \
  "$BASE/machine_events/part-00000-of-00001.csv.gz"

# expand "0-3" -> 0 1 2 3
expand() {
  for c in ${1//,/ }; do
    if [[ "$c" == *-* ]]; then
      seq "${c%-*}" "${c#*-}"
    else
      echo "$c"
    fi
  done
}

echo "==> task_events parts: $PARTS"
for p in $(expand "$PARTS"); do
  f=$(printf "%05d" "$p")
  echo "  part-$f"
  curl -L --fail --retry 3 -o "$DATA_DIR/task_events/part-$f-of-00500.csv.gz" \
    "$BASE/task_events/part-$f-of-00500.csv.gz"
done

echo "done -> $DATA_DIR"

#!/usr/bin/env bash
# PRD-262 US-050 (D4): the fp16 voice listening test, the same run in media-render.yml
# (dispatched with listening_test true) or on a machine that builds locally.
#
#   docker build -t media-render:listening services/media-render/
#   IMAGE=media-render:listening services/media-render/ci/listening_test.sh [WORK]
#
# It fetches the two fp16 candidates into WORK/kokoro (a file already there is reused),
# then speaks the script-window step's script (`fixture-bundle script`) with today's
# fp32 and each fp16, one fresh container per model, through listening_test.py speak.
# Nothing in the image changes: this folder (ci/, not in the image) and the models are
# mounted read-only, and every model file is sha256-checked inside the container before
# it loads. WORK/listening gets fp32.wav, fp16-v1.1.wav, fp16-v1.0.wav and summary.md
# (each model's voice_seconds against fp32, report only). A model that fails to load or
# speaks silence fails the run after the others have spoken. WORK defaults to
# $RUNNER_TEMP in CI, else $TMPDIR/media-render-listening.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="${1:-${RUNNER_TEMP:-${TMPDIR:-/tmp}/media-render-listening}}"
IMAGE="${IMAGE:?set IMAGE to an image built from services/media-render/, e.g. docker build -t media-render:listening services/media-render/}"
models="$work/kokoro"
out="$work/listening"

if ! docker image inspect "$IMAGE" > /dev/null; then
  echo "::error::no image $IMAGE: docker build -t $IMAGE services/media-render/"
  exit 1
fi
mkdir -p "$models" "$out"

# The two fp16 candidates; fp32 is the file the image already holds.
python3 "$here/listening_test.py" downloads > "$work/downloads.txt"
while read -r label url; do
  if [ -s "$models/$label.onnx" ]; then echo "reusing $models/$label.onnx"; continue; fi
  echo "fetching $label from $url"
  curl -fsSL --retry 3 -o "$models/$label.onnx.part" "$url"
  mv "$models/$label.onnx.part" "$models/$label.onnx"
done < "$work/downloads.txt"
chmod -R a+rX "$models"
ls -l "$models"

# The container runs as its own user: it needs to write the samples here.
chmod a+rwx "$out"
docker run --rm -e LOG_RELAY_ENABLED=false "$IMAGE" fixture-bundle script > "$out/bundle.json"
labels=$(python3 "$here/listening_test.py" labels)
# Every model gets its turn: one that fails to load or speaks silence fails the run,
# but the others' samples are still spoken and kept. A sample from an earlier run on
# this WORK is removed first, so it is never taken for this run's.
rm -f "$out/summary.md"
failed=""
for model in $labels; do
  echo "──── $model ────"
  rm -f "$out/$model.wav" "$out/$model.json"
  docker run --rm -e LOG_RELAY_ENABLED=false \
    -v "$here:/ci:ro" -v "$models:/models:ro" -v "$out:/listening" \
    --entrypoint python "$IMAGE" /ci/listening_test.py speak \
    --model "$model" --models /models --bundle /listening/bundle.json --out /listening \
    || failed="$failed $model"
done
if [ -n "$failed" ]; then echo "::error::no sample from:$failed (see each model's log above)"; exit 1; fi
python3 "$here/listening_test.py" summary --out "$out" > "$out/summary.md"
cat "$out/summary.md"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then cat "$out/summary.md" >> "$GITHUB_STEP_SUMMARY"; fi
echo "the samples: $out/fp32.wav $out/fp16-v1.1.wav $out/fp16-v1.0.wav"

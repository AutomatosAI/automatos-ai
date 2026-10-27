# syntax=docker/dockerfile:1
# voice-tts: Auto's server voice. Kokoro speaks a short text and LAME streams it
# out as MP3, for the phone (Automatos Go) and the web quick-ask. Speech only: no
# Node, no browser, no music, no video. The orchestrator reaches it at
# VOICE_TTS_URL through POST /api/voice/speech.
#
# The stages follow services/media-render/Dockerfile's (the uv stage, the Kokoro
# files fetched and checked against kokoro/SHA256SUMS, the venv installed from a
# hash-checked lock and trimmed in the RUN that installs it, the espeak data moved
# to a short real path), copied, not shared: this build context is
# services/voice-tts alone, so neither image's build reads the other's files.
# The GPL boundary: phonemizer and espeak-ng (GPL-3.0, pulled in by kokoro-onnx)
# live only in this image and media-render's. The orchestrator never imports them.
#
# Railway builds this file with no build args and no target, so the LAST stage
# is production. CI also builds `test`, which adds pytest on top of the runtime.
# No BuildKit cache mount: Railway fails one that lacks a hard-coded
# id=s/<service-id>-<path> (PRD-262 §10).

# uv from its own image, in a named stage so the tag@digest pin sits on a FROM line.
FROM ghcr.io/astral-sh/uv:0.13.4@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b AS uv

# Kokoro v1.0 (Apache-2.0 weights) from its GitHub release: the fp32 model media-render
# runs and the voices file, all 54 voices in nine languages. The build fails unless each
# matches kokoro/SHA256SUMS (the same pins as media-render's).
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS kokoro
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl; \
    rm -rf /var/lib/apt/lists/*; \
    base=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0; \
    mkdir -p /opt/kokoro; \
    curl -fsSL --retry 3 -o /opt/kokoro/kokoro-v1.0.onnx "$base/kokoro-v1.0.onnx"; \
    curl -fsSL --retry 3 -o /opt/kokoro/voices-v1.0.bin "$base/voices-v1.0.bin"
COPY kokoro/SHA256SUMS /opt/kokoro/SHA256SUMS
RUN cd /opt/kokoro && sha256sum -c SHA256SUMS && rm SHA256SUMS && chmod -R a+rX /opt/kokoro

# The Python packages, from requirements.lock (requirements.in says how to regenerate
# it): every pin with its hashes, installed exactly (--require-hashes, --no-deps) into
# /opt/venv. --compile-bytecode writes the .pyc now: the service runs as uid 1000.
# The same RUN trims the venv, as media-render's py stage does (a file deleted in a
# later layer would still be in this one):
# - onnxruntime's libonnxruntime.so.* (its Python module does not link it) and its
#   transformers, quantization and tools (never imported), numpy's tests and f2py;
# - espeakng_loader's two extra copies of libespeak-ng.so become links (the build
#   fails if they differ), and every shared library is stripped (binutils here alone);
# - espeak-ng's data is MOVED to /opt/espeak: espeak-ng truncates its data path at
#   about 160 characters, phonemizer resolves symlinks, and Kokoro takes the path from
#   VOICE_TTS_ESPEAK_DATA_PATH. Every dictionary stays: the voices speak nine languages.
# Each native part is loaded before the layer is kept.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS py
COPY --from=uv /uv /usr/local/bin/uv
COPY requirements.lock /opt/voice-tts/requirements.lock
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends binutils; \
    rm -rf /var/lib/apt/lists/*; \
    uv venv --python /usr/local/bin/python3.12 /opt/venv; \
    uv pip install --python /opt/venv/bin/python --no-cache --compile-bytecode --require-hashes --no-deps \
        -r /opt/voice-tts/requirements.lock; \
    site=/opt/venv/lib/python3.12/site-packages; \
    rm "$site"/onnxruntime/capi/libonnxruntime.so.*; \
    find "$site/numpy" -type d -name tests -prune -exec rm -rf {} +; \
    rm -r "$site"/onnxruntime/transformers "$site"/onnxruntime/quantization "$site"/onnxruntime/tools \
        "$site"/numpy/f2py /opt/venv/bin/f2py; \
    for lib in "$site"/espeakng_loader/libespeak-ng.so.*; do \
        cmp "$lib" "$site"/espeakng_loader/libespeak-ng.so; \
        ln -sf libespeak-ng.so "$lib"; \
    done; \
    find "$site" -type f \( -name '*.so' -o -name '*.so.*' \) -exec strip --strip-unneeded {} +; \
    mv "$(/opt/venv/bin/python -c 'import espeakng_loader; print(espeakng_loader.get_data_path())')" /opt/espeak; \
    test -f /opt/espeak/phontab; \
    test ! -L /opt/espeak; \
    chmod -R a+rX /opt/espeak; \
    /opt/venv/bin/python -c 'import ctypes, aiohttp, numpy, espeakng_loader; ctypes.CDLL(espeakng_loader.get_library_path()).espeak_Initialize'; \
    /opt/venv/bin/python -c 'import onnxruntime as o; print(o.get_available_providers())'

FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# lame encodes the MP3 (LGPL; it runs as its own process), tini is PID 1 and reaps
# the encoder processes. No curl: the HEALTHCHECK is Python's urllib. The build fails
# if lame is missing a library or cannot encode a second of silence.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends lame tini; \
    rm -rf /var/lib/apt/lists/*; \
    if ldd /usr/bin/lame | grep 'not found'; then echo "lame is missing a shared library" >&2; exit 1; fi; \
    head -c 48000 /dev/zero | lame --quiet -r -s 24 --bitwidth 16 --signed --little-endian -m m -b 64 --cbr \
        --resample 24 -t --noreplaygain --flush - - | head -c 2 | od -An -tx1 | grep -q 'ff f'

COPY --link --from=kokoro /opt/kokoro /opt/kokoro
COPY --link --from=py /opt/venv /opt/venv
COPY --link --from=py /opt/espeak /opt/espeak
ENV PATH=/opt/venv/bin:$PATH

RUN useradd -m -u 1000 voice
USER voice
ENV HOME=/home/voice

COPY . .
ENV PYTHONPATH=/app \
    VOICE_TTS_PORT=8100

# Python's urllib, so the runtime carries no curl: a refused connection or a status
# other than 2xx raises, and the probe fails. /healthz answers while the model loads;
# /readyz is the one that waits for it. While the container starts, Docker probes every
# 2 s (--start-interval, Engine 25+).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ['VOICE_TTS_PORT'], timeout=4)"]

EXPOSE 8100

ENTRYPOINT ["tini", "--", "python", "-m", "voice_tts"]
CMD ["serve"]

# The service's own suite runs inside the image (.github/workflows/voice-tts.yml), with
# pytest and soundfile (to decode the MP3 it checks) installed into the same venv.
FROM runtime AS test
USER root
COPY --from=uv /uv /usr/local/bin/uv
RUN uv pip install --python /opt/venv/bin/python --no-cache -r /app/requirements-test.txt
USER voice

FROM runtime AS production

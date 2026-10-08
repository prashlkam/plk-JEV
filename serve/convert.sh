#!/usr/bin/env bash
# Phase 7 — HF checkpoint -> GGUF -> quantized -> registered with Ollama.
#
# llama.cpp is built CPU-ONLY and that is deliberate, not a shortcut:
# this machine has gcc 16.2.1 and nvcc 13.4, and nvcc rejects a compiler that new,
# so a CUDA build would fail (HARDWARE.md revision 8). We only need
# `convert_hf_to_gguf.py` (pure Python) and `llama-quantize` (CPU). GPU inference is
# Ollama's job, and Ollama ships its own runtime.
set -euo pipefail

WORKSPACE="${JEV_WORKSPACE:-/var/tmp/jev-workspace}"
MODEL_DIR="${1:-$WORKSPACE/out/jev-sft/best}"
NAME="${2:-jev}"
QUANT="${3:-Q5_K_M}"
LLAMA_DIR="$WORKSPACE/build/llama.cpp"
OUT_DIR="$WORKSPACE/out/gguf"
PY="${JEV_PY:-$WORKSPACE/.venv/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== JEV -> Ollama ==========================================="
echo "  checkpoint : $MODEL_DIR"
echo "  ollama name: $NAME"
echo "  quant      : $QUANT"
echo

[ -d "$MODEL_DIR" ] || { echo "no checkpoint at $MODEL_DIR; run train/sft.py first"; exit 1; }
mkdir -p "$OUT_DIR" "$WORKSPACE/build"

# --- 1. llama.cpp (CPU only) ---------------------------------------------
if [ ! -f "$LLAMA_DIR/build/bin/llama-quantize" ]; then
  echo "--- building llama.cpp (CPU only; see HARDWARE.md rev 8) ---"
  [ -d "$LLAMA_DIR" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF >/dev/null
  cmake --build "$LLAMA_DIR/build" --target llama-quantize -j "$(nproc)"
else
  echo "--- llama.cpp already built ---"
fi

# gguf-py is needed by the converter and is not on PyPI in a matching version.
"$PY" -c "import gguf" 2>/dev/null || {
  echo "--- installing gguf-py from the llama.cpp checkout ---"
  "$PY" -m pip install --quiet -e "$LLAMA_DIR/gguf-py"
}

# --- 2. convert to f16 GGUF ----------------------------------------------
F16="$OUT_DIR/$NAME-f16.gguf"
echo
echo "--- converting to GGUF (f16) ---"
"$PY" "$LLAMA_DIR/convert_hf_to_gguf.py" "$MODEL_DIR" \
      --outfile "$F16" --outtype f16

# --- 3. quantize ----------------------------------------------------------
QF="$OUT_DIR/$NAME-${QUANT,,}.gguf"
echo
echo "--- quantizing -> $QUANT ---"
"$LLAMA_DIR/build/bin/llama-quantize" "$F16" "$QF" "$QUANT"
ls -lh "$F16" "$QF" | awk '{print "  "$5"\t"$9}'

# --- 4. Modelfile ---------------------------------------------------------
# The template is intentionally a bare passthrough: JEV is trained on a raw prompt
# built by jev.spec.render_prompt, and the SDK calls /api/generate with raw=true.
# Any chat scaffolding here would move the decision away from token 0 and
# reintroduce the position-0 corruption that spike 0.1 found.
MF="$OUT_DIR/Modelfile.$NAME"
sed -e "s|@GGUF@|$QF|g" -e "s|@QUANT@|$QUANT|g" \
    "$REPO_ROOT/serve/Modelfile.tmpl" > "$MF"
echo
echo "--- Modelfile ---"
sed 's/^/  /' "$MF"

# --- 5. register ----------------------------------------------------------
echo
echo "--- ollama create $NAME ---"
ollama create "$NAME" -f "$MF"

# --- 6. ship the spec alongside the weights -------------------------------
# Weights are NOT usable correctly without these three: a consumer applying the
# wrong temperatures gets plausible-looking, miscalibrated confidence (PLAN.md 8).
BUNDLE="$OUT_DIR/$NAME-bundle"
mkdir -p "$BUNDLE"
cp -r "$REPO_ROOT/spec/schemas" "$BUNDLE/"
cp "$REPO_ROOT/spec/label_vocab.json" "$REPO_ROOT/spec/template.json" "$BUNDLE/"
if [ -f "$REPO_ROOT/spec/calibration.json" ]; then
  cp "$REPO_ROOT/spec/calibration.json" "$BUNDLE/"
  if ! grep -q '"authoritative": true' "$BUNDLE/calibration.json"; then
    echo "  NOTE: bundled calibration is PRELIMINARY (fitted on the fp32 checkpoint)."
    echo "        It is wrong for this quantized GGUF until you run:"
    echo "          make calibrate-final      # refits, refreshes the bundle, verifies"
  fi
else
  echo "  WARNING: no spec/calibration.json — run train/calibrate.py; confidences will be UNCALIBRATED"
fi
cp "$MF" "$BUNDLE/Modelfile"
[ -f "$MODEL_DIR/jev_meta.json" ] && cp "$MODEL_DIR/jev_meta.json" "$BUNDLE/"
echo "  bundle -> $BUNDLE"

echo
echo "=== done ===================================================="
echo "verify:   python -m serve.jev_sdk.smoke --model $NAME"
echo "evaluate: python eval/harness.py --model $NAME"
echo
echo "REMINDER: temperatures fitted on the fp32 checkpoint are preliminary."
echo "Quantization perturbs logits (PLAN.md 7), so refit against this GGUF:"
echo "  python train/calibrate.py --backend ollama --model $NAME"

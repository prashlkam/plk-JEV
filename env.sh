# source env.sh — environment for every JEV command.
#
# Heavy artifacts live on / (63 GiB) rather than /home (7.2 GiB); the venv alone is
# 7.0 GiB and would have filled /home. See HARDWARE.md revision 7.
export JEV_WORKSPACE="${JEV_WORKSPACE:-/var/tmp/jev-workspace}"
export JEV_PY="$JEV_WORKSPACE/.venv/bin/python"

# HF downloads go to the workspace, not ~/.cache.
export HF_HOME="$JEV_WORKSPACE/hf-cache"

# Fragmentation mitigation. At ~3.3 GiB usable this is the difference between a
# run completing and an OOM two hours in.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Turing has no TF32 and emulated bf16; nothing to enable. fp32 is fastest here.
export TOKENIZERS_PARALLELISM=false

# Ollama must not hold VRAM while training: a resident 0.6B model costs 964 MiB of
# the 3.3 GiB available (HARDWARE.md revision 4).
jev_gpu_free() { ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}' | xargs -r -n1 ollama stop; }

alias jev-py="$JEV_PY"
echo "JEV env: workspace=$JEV_WORKSPACE python=$($JEV_PY --version 2>&1)"

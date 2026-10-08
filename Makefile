#  JEV — a System One decision model, sized for a GTX 1650.
#  `make help` for the phase-by-phase walkthrough.

WS      ?= /var/tmp/jev-workspace
PY      := $(WS)/.venv/bin/python
MODEL   ?= HuggingFaceTB/SmolLM2-135M
NAME    ?= jev
QUANT   ?= Q5_K_M
SFT_OUT ?= $(WS)/out/jev-sft
export HF_HOME := $(WS)/hf-cache
export PYTORCH_CUDA_ALLOC_CONF := expandable_segments:True
export TOKENIZERS_PARALLELISM := false

.PHONY: help setup check-gpu spikes data data-cpt baselines train-sft train-cpt \
        calibrate calibrate-final serve eval eval-hf smoke test all clean clean-all \
        gpu-free compare annotate merge-annotations verify-bundle

help:
	@echo "JEV — phases in order. See README.md, PLAN.md, HARDWARE.md."
	@echo
	@echo "  make setup           create the workspace venv and install deps"
	@echo "  make check-gpu       verify torch sees sm_75 and report the dtype ranking"
	@echo "  make spikes          re-run the three Phase 0 feasibility spikes"
	@echo "  make data            generate corpus, split, measure agreement  (~20 s)"
	@echo "  make baselines       prior + TF-IDF + multi-head encoder        (~10 m)"
	@echo "  make train-sft       Phase 5 decision SFT                     (~3.7 h)"
	@echo "  make train-cpt       Phase 4 continued pretraining (OPTIONAL)   (~8 h)"
	@echo "  make calibrate       fit temperatures on the fp32 checkpoint"
	@echo "  make serve           GGUF + quantize + register with Ollama"
	@echo "  make calibrate-final refit temperatures against the GGUF  <- ship this"
	@echo "  make eval            full harness: metrics, gates, OOD, reliability"
	@echo "  make compare         JEV vs every baseline, with a verdict"
	@echo "  make verify-bundle   refuse to ship an incoherent weights+spec bundle"
	@echo "  make smoke           SDK end-to-end against the deployed model"
	@echo "  make test            unit tests"
	@echo
	@echo "  real labels (the next step — see data/README.md):"
	@echo "    make annotate ANNOTATOR=<you>   blind-label the hard slice"
	@echo "    make merge-annotations          combine annotators -> soft targets"
	@echo "  make all             data -> baselines -> sft -> calibrate -> serve -> eval"
	@echo
	@echo "  MODEL=$(MODEL)"
	@echo "  NAME=$(NAME)  QUANT=$(QUANT)  WS=$(WS)"

setup:
	@mkdir -p $(WS)/{hf-cache,data,out,build,pip-cache}
	@test -d $(WS)/.venv || (cd $(WS) && uv venv --python 3.12 .venv)
	@ln -sfn $(WS) ./workspace
	UV_CACHE_DIR=$(WS)/pip-cache uv pip install --python $(PY) \
	  torch --torch-backend=cu126
	UV_CACHE_DIR=$(WS)/pip-cache uv pip install --python $(PY) \
	  "transformers>=4.46" "datasets>=3.0" accelerate peft trl bitsandbytes \
	  numpy scipy scikit-learn pydantic httpx sentencepiece protobuf \
	  matplotlib tensorboard pytest datasketch safetensors pyyaml
	@$(MAKE) check-gpu

check-gpu:
	@$(PY) -c "import torch,time; \
print('torch',torch.__version__,'| cuda',torch.cuda.is_available()); \
print('arch list:',torch.cuda.get_arch_list()); \
print('device:',torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0)); \
print('native bf16:',torch.cuda.is_bf16_supported(including_emulation=False)); \
f,t=torch.cuda.mem_get_info(); print(f'VRAM free {f/2**30:.2f}/{t/2**30:.2f} GiB')"
	@echo "sm_75 must appear in the arch list; if it does not, the wheel dropped Turing."

gpu-free:
	@ollama ps 2>/dev/null | tail -n +2 | awk '{print $$1}' | xargs -r -n1 ollama stop || true
	@echo "Ollama unloaded; VRAM freed for training."

spikes:
	python3 spikes/spike_0_1_format_logprobs.py
	$(PY) spikes/spike_0_2_trainstep.py --model $(MODEL)
	$(PY) spikes/spike_0_3_tokenizer.py $(MODEL)

data:
	$(PY) data/gen_seed.py --n 120000
	$(PY) data/splits.py
	$(PY) data/agreement.py
	$(PY) -m jev.spec

data-cpt:
	$(PY) data/prepare_cpt.py --model $(MODEL) --target-tokens 50000000

baselines: gpu-free
	$(PY) eval/baselines/tfidf.py
	$(PY) eval/baselines/encoder.py

train-cpt: gpu-free
	$(PY) train/cpt.py --model $(MODEL) --out $(WS)/out/jev-cpt

train-sft: gpu-free
	$(PY) train/sft.py --config train/configs/sft_smollm135.yaml --out $(SFT_OUT)

calibrate: gpu-free
	$(PY) train/calibrate.py --backend hf --model $(SFT_OUT)/best

serve:
	bash serve/convert.sh $(SFT_OUT)/best $(NAME) $(QUANT)

# Refits against the QUANTIZED GGUF, then refreshes the bundle and verifies it.
# convert.sh necessarily copies calibration.json before this step exists, so the
# bundle MUST be refreshed here — the first full run shipped stale fp32
# temperatures because it was not (see serve/verify_bundle.py).
calibrate-final:
	$(PY) train/calibrate.py --backend ollama --model $(NAME) --limit 1200
	cp spec/calibration.json $(WS)/out/gguf/$(NAME)-bundle/calibration.json
	$(PY) serve/verify_bundle.py --bundle $(WS)/out/gguf/$(NAME)-bundle

verify-bundle:
	$(PY) serve/verify_bundle.py --bundle $(WS)/out/gguf/$(NAME)-bundle

eval:
	$(PY) eval/harness.py --model $(NAME) --backend ollama
	$(PY) eval/compare.py

eval-hf: gpu-free
	$(PY) eval/harness.py --model $(SFT_OUT)/best --backend hf --no-ood

smoke:
	$(PY) -m serve.jev_sdk.smoke --model $(NAME)

compare:
	$(PY) eval/compare.py

# Blind active-learning annotation. Each annotator writes their OWN file; votes
# are only ever combined by merge-annotations, never inside the tool.
ANNOTATOR ?= $(USER)
annotate:
	$(PY) tools/annotate.py --annotator $(ANNOTATOR) --mode hard --model $(NAME)

merge-annotations:
	$(PY) tools/merge_annotations.py
	$(PY) data/agreement.py --inp $(WS)/data/sft/annotated.jsonl

test:
	$(PY) -m pytest tests/ -q

all:
	$(MAKE) data
	$(MAKE) baselines
	$(MAKE) train-sft
	$(MAKE) calibrate
	$(MAKE) serve
	$(MAKE) calibrate-final
	$(MAKE) eval
	$(MAKE) compare
	$(MAKE) smoke

clean:
	rm -rf $(WS)/out/* eval/reports/*

clean-all: clean
	rm -rf $(WS)/data/* $(WS)/build/*
	@echo "venv and hf-cache kept; remove $(WS) by hand to reclaim ~8 GiB."

"""JEV — a System One decision model: schema-constrained, calibrated, one token.

Layout:
  jev.spec     task schemas, prompt rendering, enum<->digit encoding
  jev.metrics  calibration + decision-quality metrics (numpy only)
  jev.ollama   serving client that renormalizes Ollama's pre-mask logprobs
"""
from .spec import (ABSTAIN_CODE, TaskSpec, get_spec, load_specs, render_prompt,
                   normalize_input, build_label_vocab, write_label_vocab)

__all__ = ["ABSTAIN_CODE", "TaskSpec", "get_spec", "load_specs", "render_prompt",
           "normalize_input", "build_label_vocab", "write_label_vocab"]
__version__ = "0.1.0"

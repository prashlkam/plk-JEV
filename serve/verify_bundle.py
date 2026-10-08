#!/usr/bin/env python3
"""Refuse to ship an incoherent bundle.

PLAN.md 8: the weights are not usable alone. A consumer holding the GGUF and the
wrong `calibration.json` gets plausible-looking, miscalibrated confidence — there is
no error, no warning, just quietly wrong numbers.

That failure actually happened during the first full run: `serve/convert.sh` copies
`spec/calibration.json` into the bundle, but it runs BEFORE `calibrate-final`, so
the shipped copy held the preliminary fp32 temperatures while `spec/` held the
authoritative GGUF-fitted ones. `route_v1` differed by 0.73 vs 1.08 — under- vs
over-confident, in opposite directions.

So the ordering is now checked rather than remembered:

  * the bundle's calibration must be `authoritative: true` (fitted via --backend ollama)
  * it must match `spec/calibration.json` exactly
  * schemas, label_vocab and template must match the repo
  * the label_vocab's template must match the one the model was trained on

Exit code is non-zero on any failure, so `make calibrate-final` fails loudly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev.spec import build_label_vocab, load_specs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--allow-preliminary", action="store_true",
                    help="permit non-authoritative (fp32-fitted) temperatures; "
                         "for local iteration only, never for a release")
    args = ap.parse_args()

    b = Path(args.bundle)
    problems, warnings = [], []

    if not b.is_dir():
        sys.exit(f"no bundle at {b}")

    # --- required files ------------------------------------------------
    required = ["calibration.json", "label_vocab.json", "template.json",
                "Modelfile", "schemas"]
    for f in required:
        if not (b / f).exists():
            problems.append(f"missing {f}")
    if problems:
        for p in problems:
            print(f"  FAIL  {p}")
        sys.exit(1)

    # --- calibration is authoritative and matches spec/ ----------------
    bc = json.loads((b / "calibration.json").read_text())
    sc_path = ROOT / "spec" / "calibration.json"
    if not bc.get("fitted"):
        problems.append("bundle calibration.json is not fitted — run train/calibrate.py")
    if not bc.get("authoritative"):
        msg = (f"bundle calibration was fitted with backend={bc.get('backend')!r}, "
               "not 'ollama'. Quantization perturbs logits (PLAN.md 7), so these "
               "temperatures are PRELIMINARY and will be wrong for the shipped GGUF.")
        (warnings if args.allow_preliminary else problems).append(msg)
    if sc_path.exists():
        sc = json.loads(sc_path.read_text())
        if sc.get("temperatures") != bc.get("temperatures"):
            diff = {k: (round(sc["temperatures"].get(k, float("nan")), 3),
                        round(bc["temperatures"].get(k, float("nan")), 3))
                    for k in set(sc.get("temperatures", {})) |
                             set(bc.get("temperatures", {}))
                    if sc.get("temperatures", {}).get(k)
                       != bc.get("temperatures", {}).get(k)}
            problems.append(
                f"bundle temperatures differ from spec/calibration.json: "
                f"{{task: (spec, bundle)}} = {diff}. Re-run "
                f"`make calibrate-final` to refresh the bundle.")
        if sc.get("authoritative") and not bc.get("authoritative"):
            problems.append(
                "spec/calibration.json IS authoritative but the bundle's is not — "
                "the bundle was built before calibrate-final ran.")

    # --- every task has a temperature ----------------------------------
    specs = load_specs()
    missing_T = sorted(set(specs) - set(bc.get("temperatures", {})))
    if missing_T:
        problems.append(f"no temperature for {missing_T}; those tasks would "
                        f"silently serve at T=1.0")

    # --- spec artifacts match the repo ---------------------------------
    bv = json.loads((b / "label_vocab.json").read_text())
    if bv != build_label_vocab():
        problems.append("bundle label_vocab.json does not match spec/schemas; a "
                        "consumer would decode codes to the wrong enum values")
    bt = json.loads((b / "template.json").read_text())
    st = json.loads((ROOT / "spec" / "template.json").read_text())
    if bt.get("template") != st.get("template"):
        problems.append(f"template mismatch: bundle {bt.get('template')!r} vs repo "
                        f"{st.get('template')!r}")

    # --- the template the model was TRAINED on -------------------------
    meta_p = b / "jev_meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta.get("template") and meta["template"] != st.get("template"):
            problems.append(
                f"the model was TRAINED on template {meta['template']!r} but the "
                f"bundle serves {st.get('template')!r}. Accuracy and calibration "
                f"both degrade silently.")
    else:
        warnings.append("no jev_meta.json: cannot verify the training template")

    # --- schemas ------------------------------------------------------
    for p in (ROOT / "spec" / "schemas").glob("*.json"):
        q = b / "schemas" / p.name
        if not q.exists():
            problems.append(f"missing schemas/{p.name}")
        elif json.loads(q.read_text()) != json.loads(p.read_text()):
            problems.append(f"schemas/{p.name} differs from the repo")

    # --- report -------------------------------------------------------
    print(f"bundle: {b}")
    print(f"  calibration backend : {bc.get('backend')} "
          f"(authoritative={bc.get('authoritative')})")
    print(f"  temperatures        : "
          f"{ {k: round(v, 3) for k, v in sorted(bc.get('temperatures', {}).items())} }")
    print(f"  tasks               : {len(specs)}")
    if meta_p.exists():
        m = json.loads(meta_p.read_text())
        print(f"  trained             : step {m.get('step')}, "
              f"base {m.get('base_model')}, macro_ece "
              f"{m.get('macro_ece', float('nan')):.4f}, "
              f"legal_mass {m.get('legal_mass_mean', float('nan')):.4f}")
    for w in warnings:
        print(f"  WARN  {w}")
    for p in problems:
        print(f"  FAIL  {p}")

    if problems:
        print(f"\n{len(problems)} problem(s): this bundle is NOT shippable.")
        sys.exit(1)
    print("\nshippable: calibration is authoritative and every artifact matches the repo.")


if __name__ == "__main__":
    main()

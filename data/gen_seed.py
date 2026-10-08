#!/usr/bin/env python3
"""Generate the synthetic seed corpus.

WHY THIS EXISTS, AND WHAT IT IS NOT
-----------------------------------
PLAN.md 4b ranks data sources: real production labels first, frontier distillation
second, human labeling for the hard slice. This project has none of those yet, and
waiting for them would leave every other phase untestable. So this generator stands
in — it makes the whole pipeline runnable end to end today.

It is NOT a substitute for real data, and results measured on it must never be
reported as model quality. Labels here are produced by a deterministic function of
latent facts, so a model can in principle reach ~100% — it is learning the
generator, not the task. `eval/harness.py` stamps `synthetic: true` on any report
scored against this corpus and refuses to omit the caveat.

What it IS good for:
  * proving the training, calibration, packaging and serving path works;
  * exercising soft targets and annotator disagreement with known ground truth,
    so `agreement.py` and the calibration code can be validated against a corpus
    whose true label distribution is known exactly;
  * generating a deliberately shifted OOD split to check that confidence drops.

DESIGN
------
Messages are rendered from *latent facts* (topic, impact, tone, frame, channel) and
all six labels are derived from those facts by rules that mirror spec/tasks.md.
This gives realistic correlation structure between tasks — urgency correlates with
but does not determine triage, route correlates with blocker — which is what makes
the multi-task setup non-trivial.

Ambiguity is injected deliberately, not left to chance: a configurable fraction of
samples draw latent combinations that genuinely sit on a decision boundary, and the
simulated annotators disagree on exactly those. That is what produces meaningful
soft targets instead of a corpus where every label is 1.0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev.spec import get_spec, load_specs, normalize_input  # noqa: E402

# --- latent fact space ---------------------------------------------------

TOPICS = [
    "charge", "refund", "pricing", "login", "permission", "export",
    "outage", "bug", "data_loss", "quote", "upgrade", "spam_report",
    "cancel", "praise", "feature",
]
IMPACTS = ["total", "major", "minor", "none"]
TONES = ["calm", "firm", "urgent", "furious"]
FRAMES = ["ask", "report", "demand", "thanks"]

# --- label rules (mirror spec/tasks.md) ----------------------------------

ROUTE_BY_TOPIC = {
    "charge": "billing", "refund": "billing", "pricing": "billing",
    "cancel": "billing",
    "login": "account", "permission": "account", "export": "account",
    "outage": "technical", "bug": "technical", "data_loss": "technical",
    "quote": "sales", "upgrade": "sales", "feature": "sales",
    "spam_report": "abuse",
    "praise": "technical",
}
BLOCKER_BY_TOPIC = {
    "charge": "payment", "refund": "payment", "pricing": "none",
    "cancel": "none", "login": "credentials", "permission": "permissions",
    "export": "permissions", "outage": "outage", "bug": "none",
    "data_loss": "data_loss", "quote": "none", "upgrade": "none",
    "feature": "none", "spam_report": "none", "praise": "none",
}
INTENT_BY_FRAME_TOPIC = {
    ("thanks", None): "praise",
    ("ask", "feature"): "feature_request",
    ("report", "feature"): "feature_request",
    ("demand", "feature"): "feature_request",
}


def derive_labels(f: dict) -> dict:
    """The generator's ground truth. Rules deliberately mirror spec/tasks.md."""
    topic, impact, tone, frame = f["topic"], f["impact"], f["tone"], f["frame"]

    route = ROUTE_BY_TOPIC[topic]
    # spec/tasks.md edge case: the *ask* decides, not the cause.
    if topic == "login" and f.get("caused_by_outage"):
        route = "technical"
    if topic == "charge" and f.get("after_access_issue"):
        route = "billing"

    if impact == "total" or topic == "data_loss":
        triage = "P0"
    elif impact == "major":
        triage = "P1"
    elif impact == "minor":
        triage = "P2"
    else:
        triage = "P3"
    if topic in ("quote", "upgrade", "pricing", "feature", "praise"):
        triage = "P3"

    intent = INTENT_BY_FRAME_TOPIC.get((frame, topic)) \
        or INTENT_BY_FRAME_TOPIC.get((frame, None))
    if intent is None:
        if topic == "cancel":
            intent = "cancel"
        elif topic in ("bug", "outage", "data_loss"):
            intent = "bug_report"
        elif topic == "feature":
            intent = "feature_request"
        elif frame == "demand" or tone == "furious":
            intent = "complaint"
        elif frame == "report":
            intent = "bug_report"
        else:
            intent = "question"

    blocker = BLOCKER_BY_TOPIC[topic]
    if topic == "login" and f.get("caused_by_outage"):
        blocker = "outage"

    urgency = {"calm": "1", "firm": "2", "urgent": "4", "furious": "5"}[tone]
    if tone == "calm" and impact == "total":
        urgency = "3"          # calm P0: objective severity, low expressed urgency
    if tone == "furious" and impact == "none":
        urgency = "5"          # furious P3: the inverse case
    if f.get("repeat_contact") and urgency in ("1", "2"):
        urgency = "3"

    needs_human = (
        triage == "P0" or tone == "furious" or topic in ("cancel", "spam_report",
                                                         "data_loss")
        or f.get("legal_threat") or f.get("repeat_contact")
    )
    return {
        "route_v1": route,
        "triage_v1": triage,
        "gate_human_v1": "true" if needs_human else "false",
        "intent_v1": intent,
        "blocker_v1": blocker,
        "urgency_v1": urgency,
    }


# --- surface realization -------------------------------------------------

BODY = {
    "charge": ["I was charged {amount} twice this month",
               "there are two {amount} charges on my card for the same invoice",
               "my statement shows a duplicate charge of {amount}",
               "why has {amount} been taken again"],
    "refund": ["still waiting on the refund you approved {days} days ago",
               "the refund for invoice {inv} never arrived",
               "I was told {amount} would be refunded, nothing yet"],
    "pricing": ["is the team plan priced per seat or per workspace",
                "what does the {amount} tier actually include",
                "does annual billing come with a discount"],
    "login": ["I can't log in, it keeps saying invalid credentials",
              "password reset emails never arrive",
              "locked out of my account since this morning",
              "SSO redirects me back to the login page"],
    "permission": ["I can't see the admin settings anymore",
                   "my role was changed and now half the pages 404",
                   "need write access to the shared workspace"],
    "export": ["the CSV export button does nothing",
               "I need a full export of my data for compliance",
               "export finishes but the file is empty"],
    "outage": ["the whole dashboard is returning 503",
               "nothing loads, status page says all green",
               "API has been timing out for {mins} minutes"],
    "bug": ["the date picker jumps to 1970 when I pick December",
            "saving a draft silently drops the attachments",
            "search returns results from a different project",
            "the chart axis labels overlap on mobile"],
    "data_loss": ["my last {days} days of records are gone",
                  "a project disappeared after the migration",
                  "all comments on the board were wiped"],
    "quote": ["can you send a quote for {seats} seats",
              "we need pricing for a {seats}-person team with SSO",
              "looking for an annual quote to take to procurement"],
    "upgrade": ["how do I move from starter to the team plan",
                "want to add {seats} more seats mid-cycle"],
    "feature": ["any plans to add dark mode",
                "it would help a lot if we could bulk-tag items",
                "please add webhook support for deletions"],
    "spam_report": ["a user keeps posting crypto links in the public board",
                    "reporting {handle} for harassment in comments",
                    "someone is scraping our workspace and spamming members"],
    "cancel": ["I want to cancel my subscription",
               "please close my account at the end of the cycle",
               "cancelling, we're moving to another tool"],
    "praise": ["just wanted to say the new editor is excellent",
               "your support team sorted this out fast, thank you",
               "the latest release fixed everything, nice work"],
}

OPEN = {
    "calm": ["Hi,", "Hello,", "Hi team,", "Morning,", ""],
    "firm": ["Hi,", "Hello,", "Following up:", "Again -"],
    "urgent": ["Hi,", "Urgent:", "We need help -", "Please read:"],
    "furious": ["This is unacceptable.", "I've had enough.", "Seriously?",
                "THIS IS THE THIRD TIME."],
}
CLOSE = {
    "calm": ["Thanks.", "Any pointers appreciated.", "Thanks in advance.", ""],
    "firm": ["Please advise.", "Can someone look at this today.",
             "I'd like an update.", "Please confirm."],
    "urgent": ["We're blocked on this.", "Need this resolved today.",
               "Please escalate.", "This is blocking our release."],
    "furious": ["Fix this or we're leaving.", "I want this escalated now.",
                "Cancel everything if this isn't fixed.",
                "I'm done asking nicely."],
}
IMPACT_CLAUSE = {
    "total": ["Nothing works at all.", "The entire team is blocked.",
              "Production is down.", "No one can work."],
    "major": ["There's no workaround we can find.", "This blocks our main workflow.",
              "Half the team is affected."],
    "minor": ["We found a workaround for now.", "It's annoying but survivable.",
              "Only affects one page."],
    "none": ["", "", "No rush on this."],
}
REPEAT = ["This is my third message about this.", "I've already emailed twice.",
          "Following up again on ticket {inv}."]
LEGAL = ["Our legal team is now involved.",
         "This may breach our DPA.",
         "I'm consulting our lawyer about the contract."]


#: Discourse markers that demote a clause to a *resolved* or *secondary* mention.
#: `--hard` uses these to force composition: the message mentions two topics, and
#: only the marker says which one is the live ask.
DEMOTE_PRE = [
    "The {other} issue from last month is sorted, but",
    "Separately from the {other} thing which is fine now,",
    "Not about the {other} this time -",
    "Ignore my earlier {other} message, that's resolved.",
]
DEMOTE_POST = [
    "(The {other} question from before is already answered, no need to revisit.)",
    "Unrelated: the {other} thing resolved itself, so nothing needed there.",
    "I also raised a {other} ticket earlier but that one's closed.",
]
OTHER_WORD = {
    "charge": "billing", "refund": "refund", "pricing": "pricing",
    "login": "login", "permission": "permissions", "export": "export",
    "outage": "outage", "bug": "bug", "data_loss": "data", "quote": "quote",
    "upgrade": "upgrade", "spam_report": "abuse", "cancel": "cancellation",
    "praise": "feedback", "feature": "feature",
}


def add_distractor(f: dict, primary: str, rng: random.Random, fill: dict) -> str:
    """Mention a second topic lexically, but mark it as not the live ask.

    This is what makes the corpus non-trivial. Without it, each topic's body
    template is a unique lexical fingerprint and TF-IDF + logistic regression
    scores 1.000 on five of six tasks — measured, see eval/reports/baseline_tfidf.json
    for the easy mode. With it, both topics' vocabulary is present and the label
    depends on discourse structure rather than word presence.
    """
    others = [t for t in TOPICS if t != f["topic"]]
    other = rng.choice(others)
    other_body = rng.choice(BODY[other]).format(**fill).rstrip(".")
    word = OTHER_WORD[other]
    if rng.random() < 0.5:
        pre = rng.choice(DEMOTE_PRE).format(other=word)
        return f"{pre} {other_body} — anyway. {primary}"
    post = rng.choice(DEMOTE_POST).format(other=word)
    return f"{primary} {post} Also saw that {other_body}."


def render(f: dict, rng: random.Random, hard: bool = False) -> str:
    fill = {
        "amount": rng.choice(["$19", "$49", "$120", "$1,400", "£89", "€250"]),
        "days": rng.choice(["3", "5", "10", "14"]),
        "mins": rng.choice(["20", "45", "90"]),
        "inv": f"INV-{rng.randint(10000, 99999)}",
        "seats": rng.choice(["12", "25", "40", "120"]),
        "handle": rng.choice(["@tradebot99", "@freecoins", "@user4412"]),
    }
    parts = [rng.choice(OPEN[f["tone"]]),
             rng.choice(BODY[f["topic"]]).format(**fill) + "."]
    if f["topic"] not in ("praise", "quote", "pricing", "upgrade", "feature"):
        c = rng.choice(IMPACT_CLAUSE[f["impact"]])
        if c:
            parts.append(c)
    if f.get("repeat_contact"):
        parts.append(rng.choice(REPEAT).format(**fill))
    if f.get("legal_threat"):
        parts.append(rng.choice(LEGAL))
    if f.get("caused_by_outage"):
        parts.append("Started right after the outage this morning.")
    parts.append(rng.choice(CLOSE[f["tone"]]))
    txt = " ".join(p for p in parts if p).strip()
    if hard and rng.random() < 0.55:
        txt = add_distractor(f, txt, rng, fill)
    if f["tone"] == "furious" and rng.random() < 0.3:
        txt = txt.upper()
    return normalize_input(txt)


# --- ambiguity -----------------------------------------------------------

#: Latent combinations that genuinely sit on a decision boundary. Annotators
#: disagree here on the named task, producing soft targets with known structure.
AMBIGUOUS = [
    ({"topic": "login", "caused_by_outage": True}, "route_v1", ["technical", "account"]),
    ({"topic": "charge", "after_access_issue": True}, "route_v1", ["billing", "account"]),
    ({"topic": "cancel"}, "route_v1", ["billing", "account"]),
    ({"topic": "cancel"}, "intent_v1", ["cancel", "complaint"]),
    ({"topic": "bug", "impact": "major"}, "triage_v1", ["P1", "P2"]),
    ({"topic": "export", "impact": "minor"}, "blocker_v1", ["permissions", "none"]),
    ({"tone": "firm"}, "urgency_v1", ["2", "3"]),
    ({"topic": "feature", "tone": "furious"}, "intent_v1", ["feature_request", "complaint"]),
]


def sample_facts(rng: random.Random, ambiguous_rate: float) -> tuple[dict, list]:
    f = {
        "topic": rng.choice(TOPICS),
        "impact": rng.choices(IMPACTS, weights=[0.07, 0.2, 0.38, 0.35])[0],
        "tone": rng.choices(TONES, weights=[0.38, 0.3, 0.2, 0.12])[0],
        "frame": rng.choices(FRAMES, weights=[0.4, 0.33, 0.19, 0.08])[0],
        "repeat_contact": rng.random() < 0.12,
        "legal_threat": rng.random() < 0.04,
        "caused_by_outage": False,
        "after_access_issue": False,
    }
    hard = []
    if rng.random() < ambiguous_rate:
        patch, task, options = rng.choice(AMBIGUOUS)
        f.update(patch)
        hard.append({"task": task, "options": options})
    if f["topic"] == "praise":
        f["frame"], f["tone"], f["impact"] = "thanks", "calm", "none"
    if f["topic"] == "login" and rng.random() < 0.18:
        f["caused_by_outage"] = True
    return f, hard


# --- simulated annotators ------------------------------------------------

def annotate(labels: dict, hard: list, rng: random.Random, n_annotators: int,
             base_noise: float) -> dict:
    """Simulate independent annotators, then keep the disagreement.

    PLAN.md 4b: 2-of-3 becomes a 0.67 soft target, never a collapsed majority
    vote. Here disagreement is concentrated on the `hard` cases so the soft
    targets carry signal rather than uniform label noise.
    """
    hard_by_task = {h["task"]: h["options"] for h in hard}
    out = {}
    for task_id, gold in labels.items():
        spec = get_spec(task_id)
        votes = []
        for _ in range(n_annotators):
            if task_id in hard_by_task and rng.random() < 0.55:
                votes.append(rng.choice(hard_by_task[task_id]))
            elif rng.random() < base_noise:
                alt = [v for v in spec.values if v != gold]
                votes.append(rng.choice(alt) if alt else gold)
            else:
                votes.append(gold)
        counts: dict[str, int] = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
        out[task_id] = {
            "gold": gold,
            "votes": votes,
            "soft": {k: round(c / len(votes), 6) for k, c in sorted(counts.items())},
            "majority": max(counts, key=lambda k: (counts[k], k)),
            "unanimous": len(counts) == 1,
        }
    return out


# --- OOD shift -----------------------------------------------------------

def to_ood(text: str, rng: random.Random) -> str:
    """Deliberate distribution shift for the `ood` split.

    PLAN.md 4c: this measures whether confidence correctly *drops*. A model that
    is confidently wrong off-distribution is worse than useless in an automated
    pipeline, and in-distribution metrics cannot detect that.
    """
    mode = rng.choice(["terse", "logdump", "verbose", "translatese", "typos"])
    if mode == "terse":
        words = text.split()
        return " ".join(words[: max(3, len(words) // 4)]).lower()
    if mode == "logdump":
        return (f"[{rng.randint(10, 23)}:{rng.randint(10, 59)}:02] ERR svc=api "
                f"code={rng.choice([500, 502, 503, 403, 401])} msg=\"{text[:70]}\" "
                f"trace={hashlib.md5(text.encode()).hexdigest()[:12]}")
    if mode == "verbose":
        return ("To whom it may concern, I hope this message finds you well. "
                "I am writing to you today in connection with an issue that has "
                f"arisen in our use of your platform. {text} I would be most "
                "grateful for your attention to this matter at your earliest "
                "possible convenience. Yours faithfully.")
    if mode == "translatese":
        return (text.replace("I can't", "I am not able to")
                    .replace("doesn't", "does not make")
                    .replace("the ", "this ") + " Please to advise quickly.")
    chars = list(text)
    for _ in range(max(1, len(chars) // 25)):
        i = rng.randrange(len(chars))
        chars[i] = rng.choice("aeiostnr")
    return "".join(chars)


# --- main ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120_000)
    ap.add_argument("--ood-n", type=int, default=4_000)
    ap.add_argument("--annotators", type=int, default=3)
    ap.add_argument("--annotated-n", type=int, default=6_000,
                    help="size of the multi-annotator subset (PLAN.md 4b wants >=2000)")
    ap.add_argument("--ambiguous-rate", type=float, default=0.18)
    ap.add_argument("--base-noise", type=float, default=0.04)
    ap.add_argument("--hard", action="store_true", default=True,
                    help="add cross-topic distractors so labels need composition, "
                         "not lexical lookup. Off = TF-IDF scores 1.000 (measured).")
    ap.add_argument("--easy", dest="hard", action="store_false")
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--out", default="workspace/data/sft/seed.jsonl")
    ap.add_argument("--ood-out", default="workspace/data/sft/seed_ood.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tasks = list(load_specs())
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    n_written = 0
    with outp.open("w") as fh:
        while n_written < args.n:
            f, hard = sample_facts(rng, args.ambiguous_rate)
            text = render(f, rng, args.hard)
            h = hashlib.sha1(text.encode()).hexdigest()
            if h in seen:
                continue           # exact-duplicate guard; near-dupes handled in prepare
            seen.add(h)
            labels = derive_labels(f)
            rec = {
                "id": h[:16],
                "input": text,
                "latent": f,
                "labels": labels,
                "hard": hard,
                "source": "synthetic_seed",
                "hard": args.hard,
            }
            if n_written < args.annotated_n:
                rec["annotations"] = annotate(labels, hard, rng,
                                              args.annotators, args.base_noise)
            fh.write(json.dumps(rec) + "\n")
            n_written += 1

    with Path(args.ood_out).open("w") as fh:
        for _ in range(args.ood_n):
            f, hard = sample_facts(rng, args.ambiguous_rate)
            text = to_ood(render(f, rng, args.hard), rng)
            fh.write(json.dumps({
                "id": hashlib.sha1(text.encode()).hexdigest()[:16],
                "input": normalize_input(text),
                "latent": f,
                "labels": derive_labels(f),
                "hard": hard,
                "source": "synthetic_ood",
            }) + "\n")

    print(f"wrote {n_written:,} -> {outp}")
    print(f"      {args.ood_n:,} -> {args.ood_out}")
    print(f"      {min(args.annotated_n, n_written):,} carry {args.annotators}-annotator "
          f"votes and soft targets")
    print(f"      {len(tasks)} tasks per example = "
          f"{n_written * len(tasks):,} (input, task, label) training pairs")
    print("\nNOTE: synthetic. Labels are a deterministic function of latent facts, so "
          "accuracy on this corpus measures\n      pipeline correctness, not model "
          "quality. See data/README.md.")


if __name__ == "__main__":
    main()

"""
Build a supervised fine-tuning (SFT) dataset for the tactical-reasoning LLM.

Strategy (teacher-distillation + rule-based label cleaning):
  1. Sample many diverse swarm scenarios (formation pairs, speeds, stabilities,
     approach rates, drone counts, noise levels).
  2. For each, synthesise the SAME tactical-context + key-window prompt the
     real pipeline produces (so the fine-tuned model sees in-distribution input).
  3. Get a GOLD assessment. By default we call a strong TEACHER model (Groq
     llama-3.3-70b) to draft it, then OVERRIDE threat_level / likely_intent /
     recommended_action with the canonical domain rule for that scenario. This
     removes teacher noise and makes the targets internally consistent.
  4. Write chat-format JSONL: [{"role":"user",...},{"role":"assistant",...}].

Why distill + clean rather than pure rules: the teacher supplies fluent,
varied `situation_summary` / `threat_reasoning` / `follow_up_watch` text, while
the rules guarantee the decision fields are correct and consistent. That is the
behaviour we actually want the small model to learn.

Run WITHOUT a teacher (--no-teacher) to get rule-only templated targets — lower
quality prose but zero API cost, useful for a first smoke test.

Usage:
    export GROQ_API_KEY=...
    python llm_finetuning/build_sft_dataset.py --n 600 --out data/sft_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

# Make the src/ package importable when run from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from swarm_intent.config import BASE_FORMATIONS
from swarm_intent.inference import build_llm_prompt, OUTPUT_SCHEMA  # noqa: F401

# ---------------------------------------------------------------------------
# Canonical domain rules: scenario -> (threat, intent, action). This is the
# ground-truth decision logic the model must internalise. EDIT THESE with your
# domain expert — they define what "correct" means for the whole project.
# ---------------------------------------------------------------------------
RULES = {
    # (from, to) : (threat_level, likely_intent, recommended_action)

    # --- steady-state ---
    ("v_shape",      "v_shape"):      ("medium",   "surveillance",        "increase_surveillance"),
    ("encirclement", "encirclement"): ("high",     "encircle",            "alert_operator"),
    ("column",       "column"):       ("low",      "patrol",              "monitor"),
    ("diamond",      "diamond"):      ("low",      "patrol",              "monitor"),
    ("dispersed",    "dispersed"):    ("low",      "surveillance",        "monitor"),
    ("converging",   "converging"):   ("high",     "approach",            "alert_operator"),
    ("shield",       "shield"):       ("medium",   "defensive",           "monitor"),

    # --- to converging ---
    ("v_shape",      "converging"):   ("high",     "approach",            "alert_operator"),
    ("encirclement", "converging"):   ("critical", "encircle",            "deploy_countermeasure"),
    ("column",       "converging"):   ("high",     "approach",            "alert_operator"),
    ("diamond",      "converging"):   ("high",     "approach",            "alert_operator"),
    ("dispersed",    "converging"):   ("high",     "approach",            "alert_operator"),
    ("shield",       "converging"):   ("medium",   "approach",            "increase_surveillance"),

    # --- to encirclement ---
    ("v_shape",      "encirclement"): ("high",     "encircle",            "alert_operator"),
    ("column",       "encirclement"): ("high",     "encircle",            "alert_operator"),
    ("diamond",      "encirclement"): ("high",     "encircle",            "alert_operator"),
    ("dispersed",    "encirclement"): ("high",     "encircle",            "alert_operator"),
    ("converging",   "encirclement"): ("critical", "encircle",            "deploy_countermeasure"),
    ("shield",       "encirclement"): ("medium",   "encircle",            "increase_surveillance"),

    # --- from encirclement (de-escalating) ---
    ("encirclement", "v_shape"):      ("medium",   "regroup",             "increase_surveillance"),
    ("encirclement", "column"):       ("low",      "withdraw",            "monitor"),
    ("encirclement", "diamond"):      ("medium",   "consolidate",         "monitor"),
    ("encirclement", "dispersed"):    ("low",      "withdraw",            "monitor"),
    ("encirclement", "shield"):       ("medium",   "defensive",           "monitor"),

    # --- from converging (de-escalating) ---
    ("converging",   "v_shape"):      ("medium",   "regroup",             "increase_surveillance"),
    ("converging",   "column"):       ("low",      "patrol",              "monitor"),
    ("converging",   "diamond"):      ("medium",   "consolidate",         "monitor"),
    ("converging",   "dispersed"):    ("low",      "withdraw",            "monitor"),
    ("converging",   "shield"):       ("medium",   "defensive",           "monitor"),

    # --- v_shape remaining ---
    ("v_shape",      "column"):       ("low",      "transit",             "monitor"),
    ("v_shape",      "diamond"):      ("medium",   "defensive_transition","monitor"),
    ("v_shape",      "dispersed"):    ("low",      "area_search",         "monitor"),
    ("v_shape",      "shield"):       ("medium",   "defensive_transition","monitor"),

    # --- column remaining ---
    ("column",       "v_shape"):      ("high",     "attack_preparation",  "alert_operator"),
    ("column",       "diamond"):      ("medium",   "consolidate",         "monitor"),
    ("column",       "dispersed"):    ("low",      "area_search",         "monitor"),
    ("column",       "shield"):       ("medium",   "defensive_transition","monitor"),

    # --- diamond remaining ---
    ("diamond",      "v_shape"):      ("medium",   "reposition",          "increase_surveillance"),
    ("diamond",      "column"):       ("low",      "transit",             "monitor"),
    ("diamond",      "dispersed"):    ("medium",   "area_search",         "monitor"),
    ("diamond",      "shield"):       ("medium",   "defensive_transition","monitor"),

    # --- dispersed remaining ---
    ("dispersed",    "v_shape"):      ("high",     "rally",               "alert_operator"),
    ("dispersed",    "column"):       ("low",      "transit",             "monitor"),
    ("dispersed",    "diamond"):      ("medium",   "consolidate",         "increase_surveillance"),
    ("dispersed",    "shield"):       ("medium",   "defensive_transition","monitor"),

    # --- shield remaining ---
    ("shield",       "v_shape"):      ("medium",   "reposition",          "increase_surveillance"),
    ("shield",       "column"):       ("medium",   "reposition",          "monitor"),
    ("shield",       "diamond"):      ("medium",   "defensive",           "monitor"),
    ("shield",       "dispersed"):    ("low",      "surveillance",        "monitor"),
}
DEFAULT_RULE = ("medium", "reposition", "monitor")  # ponytail: safety net only, should never fire now


def synth_context(form_a: str, form_b: str, rng: random.Random) -> tuple[str, list]:
    """Fabricate a realistic tactical-context string + key windows for a scenario.

    Independent of a trained checkpoint so you can build SFT data immediately.
    """
    transitioning = form_a != form_b
    mean_conf = round(rng.uniform(0.7, 0.98), 2)
    mean_stab = round(rng.uniform(0.6, 0.95), 2)
    approach = round(rng.uniform(-1.5, 0.5), 3)
    delta_v = round(rng.uniform(-1.0, 2.0), 2)
    vel_trend = "accelerating" if delta_v > 0.5 else "decelerating" if delta_v < -0.5 else "steady"
    dominant = form_a
    history = f"{form_a} -> transitioning -> {form_b}" if transitioning else form_a

    ctx = "\n".join([
        f"Observation window: 0.0s - 60.0s (9 overlapping windows)",
        f"Dominant formation: {dominant}",
        f"Formation history: {history}",
        (f"Transition detected at t=20.0s: {form_a} -> {form_b}"
         if transitioning else "No formation transitions detected."),
        f"Velocity trend: {vel_trend} (delta_v={delta_v:+.2f})",
        f"Formation stability: {'holding' if mean_stab > 0.7 else 'degrading'} (mean={mean_stab:.2f})",
        f"Spread dynamics: {'converging (drones closing in)' if approach < -0.1 else 'stable spread'} "
        f"(mean approach_rate={approach:.3f})",
        f"Classifier confidence: mean={mean_conf:.2f}",
    ])
    key_windows = [
        {"t": "0.0-25.0s", "formation": form_a, "confidence": mean_conf,
         "velocity": round(rng.uniform(0.0, 0.1), 3), "approach": approach,
         "stability": mean_stab, "from": None, "to": None},
        {"t": "35.0-60.0s", "formation": "transitioning" if transitioning else form_a,
         "confidence": mean_conf, "velocity": round(rng.uniform(0.0, 0.1), 3),
         "approach": approach, "stability": mean_stab,
         "from": form_a if transitioning else None,
         "to": form_b if transitioning else None},
    ]
    return ctx, key_windows


def gold_assessment(form_a, form_b, ctx, teacher=None, prompt=None) -> tuple[dict, bool]:
    """Returns (assessment, used_teacher). The teacher must see the same
    schema-bearing prompt the student trains on — otherwise its JSON comes
    back with different keys and every draft.get() silently falls back to
    the template (the bug that produced a fully-templated v2 dataset)."""
    threat, intent, action = RULES.get((form_a, form_b), DEFAULT_RULE)
    draft = {}
    if teacher is not None:
        base = prompt or f"Given this UAV swarm tactical context, write the JSON assessment.\n\n{ctx}"
        # Only the teacher sees the canonical rule labels (the student prompt in
        # the dataset stays label-free). Without this, a teacher that disagrees
        # with RULES writes threat_reasoning arguing for its own prediction,
        # which then contradicts the overridden threat_level in the saved row.
        teacher_prompt = (
            f"{base}\n\n"
            f"GROUND TRUTH from the canonical rule engine — use exactly these values, "
            f'do not deviate: threat_level="{threat}", likely_intent="{intent}", '
            f'recommended_action="{action}". Write situation_summary, threat_reasoning, '
            f"key_indicators and follow_up_watch so they genuinely support this assessment."
        )
        draft = teacher.complete(teacher_prompt)
        if not isinstance(draft, dict) or "error" in draft:
            draft = {}
    # Override the decision fields with the canonical rule (clean labels).
    used_teacher = bool(draft.get("situation_summary"))
    return {
        "situation_summary": draft.get("situation_summary",
            f"The swarm is in a {form_a} formation"
            + (f" transitioning to {form_b}." if form_a != form_b else ", holding steady.")),
        "threat_level": threat,
        "threat_reasoning": draft.get("threat_reasoning",
            f"{form_a}->{form_b} dynamics with the observed approach rate indicate a {threat} threat."),
        "likely_intent": intent,
        "recommended_action": action,
        "confidence_in_assessment": "high",
        "key_indicators": draft.get("key_indicators",
            [f"{form_a}->{form_b} transition", "approach rate", "classifier confidence"]),
        "follow_up_watch": draft.get("follow_up_watch",
            "Monitor formation and approach rate over the next window."),
    }, used_teacher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", default="data/sft_train.jsonl")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--no-teacher", action="store_true",
                    help="skip the Groq teacher; use templated targets only")
    ap.add_argument("--teacher-model", default=None,
                    help="Groq model for the teacher (default: GroqClient's default). "
                         "Daily quotas are per-model, so switching models resets the budget.")
    ap.add_argument("--append", action="store_true",
                    help="add to the existing --out dataset instead of overwriting it "
                         "(loads train+val, re-shuffles, re-splits). For accumulating "
                         "teacher rows across daily quota windows / teacher models.")
    ap.add_argument("--teacher-only", action="store_true",
                    help="drop rows where the teacher fell back to templated prose")
    ap.add_argument("--max-teacher-fails", type=int, default=20,
                    help="stop after this many consecutive teacher fallbacks (daily "
                         "quota exhausted) instead of generating template filler; 0 = off")
    ap.add_argument("--seed", type=int, default=None,
                    help="default 42, but randomized under --append so re-runs don't "
                         "regenerate duplicate scenarios")
    args = ap.parse_args()
    if args.seed is None:
        args.seed = 42 if not args.append else random.SystemRandom().randrange(1 << 30)

    teacher = None
    if not args.no_teacher:
        from swarm_intent.llm.client import GroqClient
        # max_tokens=3072: reasoning teachers (qwen3 family) spend hundreds of
        # <think> tokens before the JSON; 1024 truncates mid-answer. Harmless
        # cap for non-reasoning teachers, which finish well under it.
        teacher = (GroqClient(model=args.teacher_model, max_tokens=3072)
                   if args.teacher_model
                   else GroqClient(max_tokens=3072))  # reads GROQ_API_KEY

    rng = random.Random(args.seed)
    pairs = list(RULES.keys()) + [(a, b) for a in BASE_FORMATIONS for b in BASE_FORMATIONS]
    rows = []
    n_teacher_ok = 0
    n_gen = 0
    consec_fails = 0
    t0 = time.monotonic()
    for _ in range(args.n):
        form_a, form_b = rng.choice(pairs)
        ctx, key_windows = synth_context(form_a, form_b, rng)
        # Build the exact prompt the real pipeline uses.
        prompt = build_llm_prompt(
            predictions=[{**kw, "time_start_s": 0, "time_end_s": 0,
                          "formation_type": kw["formation"], "centroid_velocity": kw["velocity"],
                          "approach_rate": kw["approach"], "formation_stability": kw["stability"],
                          "formation_confidence": kw["confidence"], "role_differentiation": False,
                          "transition_from": kw["from"], "transition_to": kw["to"]} for kw in key_windows],
            tactical_context=ctx, summary={})
        gold, used_teacher = gold_assessment(form_a, form_b, ctx, teacher, prompt=prompt)
        n_gen += 1
        n_teacher_ok += used_teacher
        if teacher is not None:
            consec_fails = 0 if used_teacher else consec_fails + 1
            if args.max_teacher_fails and consec_fails >= args.max_teacher_fails:
                print(f"STOPPING EARLY: {consec_fails} consecutive teacher fallbacks — "
                      "daily token quota likely exhausted. Re-run later with --append, "
                      "or switch quota pools with --teacher-model.", flush=True)
                break
        if args.teacher_only and not used_teacher:
            continue
        rows.append({"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(gold, indent=2)},
        ]})
        if n_gen % 25 == 0:
            elapsed = time.monotonic() - t0
            eta = elapsed / n_gen * (args.n - n_gen)
            print(f"{n_gen}/{args.n} generated ({len(rows)} kept, {n_teacher_ok} with "
                  f"teacher prose) [{elapsed/60:.1f} min elapsed, ~{eta/60:.1f} min left]",
                  flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    val_path = args.out.replace(".jsonl", "_val.jsonl")

    if args.append:
        def load_rows(path):
            if not os.path.exists(path):
                return []
            with open(path) as f:
                return [json.loads(line) for line in f if line.strip()]
        existing = load_rows(args.out) + load_rows(val_path)
        print(f"--append: {len(existing)} existing + {len(rows)} new rows")
        rows = existing + rows
        # same scenario generated twice (e.g. an accidental same-seed re-run)
        # would leak between train and val after the re-split — drop dupes
        seen, uniq = set(), []
        for r in rows:
            key = r["messages"][0]["content"]
            if key not in seen:
                seen.add(key)
                uniq.append(r)
        if len(uniq) < len(rows):
            print(f"  dropped {len(rows) - len(uniq)} duplicate scenarios")
        rows = uniq

    random.Random(args.seed).shuffle(rows)
    n_val = int(len(rows) * args.val_frac)
    val, train = rows[:n_val], rows[n_val:]
    with open(args.out, "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open(val_path, "w") as f:
        for r in val:
            f.write(json.dumps(r) + "\n")
    total = time.monotonic() - t0
    print(f"Wrote {len(train)} train -> {args.out} and {len(val)} val -> {val_path} "
          f"in {total/60:.1f} min ({total/max(n_gen, 1):.1f} s/example)")
    if teacher is not None:
        pct = 100.0 * n_teacher_ok / max(n_gen, 1)
        print(f"Teacher prose used in {n_teacher_ok}/{n_gen} generated examples ({pct:.0f}%)")
        if pct < 50 and not args.teacher_only:
            print("WARNING: most examples fell back to templated prose — "
                  "check the teacher model/key before training on this file.")


if __name__ == "__main__":
    main()

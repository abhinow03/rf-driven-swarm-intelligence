"""
AUDIT.md V5 Phase 1 step 3: targeted regeneration of the 3,893 template-fallback
rows found by check_rationale_entailment.py's sibling audit (step 1/step 2 in
V5_LOG.md) -- user chose "regenerate with a fresh key" over "keep and tag" or
"drop" after seeing the distinct-summary hard fail.

Does NOT regenerate trajectories/pairs (the stratification is already fixed and
correct) -- only re-asks the teacher for the SAME (form_a, form_b, student
prompt) that was already generated, using build_teacher_prompt(..., prompt=
<existing user message content>). Since build_teacher_prompt's `base` becomes
`prompt` whenever prompt is truthy, ctx is never read on that path -- form_a/
form_b are the only extra facts needed, pulled from the ctx text via
swarm_intent.coverage's existing (form_a, form_b) extraction regex (same
phrasing synth_context() always produces).

Rows are updated IN PLACE (same prompt, replaced assistant content) -- no
re-shuffle, no re-split, no dedup needed, unlike build_sft_dataset.py's
--append path.

Usage:
    export NVIDIA_API_KEY=...
    python llm_finetuning/regenerate_fallback_rows.py --concurrency 24 --max-teacher-fails 100
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from swarm_intent.coverage import _extract_pair_from_ctx  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from build_sft_dataset import build_teacher_prompt, finalize_assessment  # noqa: E402

TEMPLATE_FOLLOWUP = "Monitor formation and approach rate over the next window."

FILES = {
    "train": "data/sft_train_v5_phase1.jsonl",
    "val": "data/sft_train_v5_phase1_val.jsonl",
    "mining": "data/sft_train_v5_phase1_mining.jsonl",
}
PROVENANCE_PATH = "data/sft_train_v5_phase1_provenance.json"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def extract_pair(user_content: str):
    ctx_start = user_content.find("--- TACTICAL CONTEXT ---")
    ctx = user_content[ctx_start:] if ctx_start >= 0 else user_content
    return _extract_pair_from_ctx(ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--max-teacher-fails", type=int, default=100)
    ap.add_argument("--teacher-model", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=1,
                     help="write progress to disk every N chunks")
    ap.add_argument("--limit", type=int, default=None,
                     help="only regenerate the first N fallback rows found (smoke testing)")
    args = ap.parse_args()

    from swarm_intent.llm.client import NvidiaClient
    teacher = (NvidiaClient(model=args.teacher_model, max_tokens=3072)
               if args.teacher_model else NvidiaClient(max_tokens=3072))

    data = {split: load_jsonl(path) for split, path in FILES.items()}
    with open(PROVENANCE_PATH) as f:
        provenance = json.load(f)
    prov_rows = provenance["rows"]

    targets = []  # (split, idx, form_a, form_b, hash)
    skipped_no_pair = 0
    for split, rows in data.items():
        for idx, r in enumerate(rows):
            prompt = r["messages"][0]["content"]
            h = hashlib.sha256(prompt.encode()).hexdigest()
            info = prov_rows.get(h)
            if info is None or info.get("used_teacher"):
                continue
            pair = extract_pair(prompt)
            if pair is None:
                skipped_no_pair += 1
                continue
            targets.append((split, idx, pair[0], pair[1], h))

    print(f"targets: {len(targets)} fallback rows to regenerate "
          f"({skipped_no_pair} skipped, no extractable (from,to) pair)", flush=True)
    if args.limit:
        targets = targets[:args.limit]
        print(f"--limit {args.limit}: only regenerating the first {len(targets)}", flush=True)

    n_gen = 0
    n_fixed = 0
    consec_fails = 0
    stop_early = False
    chunk_size = max(1, args.concurrency)
    t0 = time.monotonic()

    for chunk_start in range(0, len(targets), chunk_size):
        chunk = targets[chunk_start:chunk_start + chunk_size]
        prompts = [
            build_teacher_prompt(fa, fb, "", data[split][idx]["messages"][0]["content"])
            for split, idx, fa, fb, h in chunk
        ]
        drafts = teacher.complete_batch(prompts, batch_size=chunk_size)

        for (split, idx, fa, fb, h), draft in zip(chunk, drafts):
            gold, used_teacher = finalize_assessment(fa, fb, draft)
            n_gen += 1
            if used_teacher:
                data[split][idx]["messages"][1]["content"] = json.dumps(gold, indent=2)
                prov_rows[h]["used_teacher"] = True
                n_fixed += 1
                consec_fails = 0
            else:
                consec_fails += 1
                if args.max_teacher_fails and consec_fails >= args.max_teacher_fails:
                    print(f"STOPPING EARLY: {consec_fails} consecutive teacher fallbacks -- "
                          "quota likely exhausted. Re-run this script later; it will "
                          "automatically resume from the still-unfixed rows.", flush=True)
                    stop_early = True
                    break

        elapsed = time.monotonic() - t0
        eta = elapsed / n_gen * (len(targets) - n_gen) if n_gen else 0.0
        print(f"{n_gen}/{len(targets)} regenerated ({n_fixed} fixed) "
              f"[{elapsed/60:.1f} min elapsed, ~{eta/60:.1f} min left]", flush=True)

        chunk_idx = chunk_start // chunk_size
        if (chunk_idx + 1) % args.checkpoint_every == 0 or stop_early:
            for split, path in FILES.items():
                write_jsonl(path, data[split])
            with open(PROVENANCE_PATH, "w") as f:
                json.dump(provenance, f, indent=0)

        if stop_early:
            break

    for split, path in FILES.items():
        write_jsonl(path, data[split])
    with open(PROVENANCE_PATH, "w") as f:
        json.dump(provenance, f, indent=0)

    print(f"DONE: {n_fixed}/{len(targets)} rows fixed this run.", flush=True)
    remaining = sum(1 for v in prov_rows.values() if not v.get("used_teacher"))
    print(f"remaining fallback rows across corpus: {remaining}/{len(prov_rows)}", flush=True)


if __name__ == "__main__":
    main()

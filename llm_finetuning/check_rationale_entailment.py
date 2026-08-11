import json, re

EXCLUDE_WORDS = {"confidence", "stability", "speed", "velocity", "spread", "approach",
                  "proximity", "altitude", "closure", "risk", "concern", "probability",
                  "likelihood", "value", "rate"}
WINDOW = 3  # tokens on either side -- tuned on mining split (see V5_LOG.md step 2)

def filtered_matches(text):
    tokens = re.findall(r"[A-Za-z']+", text)
    lower = [t.lower() for t in tokens]
    kept = []
    for i, t in enumerate(lower):
        if t in ("low", "medium", "high", "critical"):
            ctx = lower[max(0, i-WINDOW):i] + lower[i+1:i+1+WINDOW]
            if any(c in EXCLUDE_WORDS for c in ctx):
                continue
            kept.append(t)
    return kept

def check_row(threat_level, threat_reasoning):
    kept = filtered_matches(threat_reasoning)
    other = set(kept) - {threat_level}
    if not kept:
        return "no_signal"
    if other:
        return "contradiction"
    return "entailed"

if __name__ == "__main__":
    with open('data/sft_train_v5_phase1_mining.jsonl') as f:
        mining = [json.loads(l) for l in f]
    TEMPLATE_FOLLOWUP = 'Monitor formation and approach rate over the next window.'
    teacher_rows = [r for r in mining if json.loads(r['messages'][1]['content']).get('follow_up_watch') != TEMPLATE_FOLLOWUP]

    from collections import Counter
    results = Counter()
    contradictions = []
    for r in teacher_rows:
        a = json.loads(r['messages'][1]['content'])
        verdict = check_row(a['threat_level'], a['threat_reasoning'])
        results[verdict] += 1
        if verdict == "contradiction":
            contradictions.append(a)

    print("mining teacher rows:", len(teacher_rows))
    print(results)
    print(f"precision-relevant: {len(contradictions)} flagged, {len(contradictions)/len(teacher_rows):.1%} of teacher rows")
    print()
    print("=== ALL flagged contradictions (manual precision check) ===")
    for a in contradictions:
        print(f"threat_level={a['threat_level']}  |  {a['threat_reasoning']}")
        print()

# AUDIT.md / V5_LOG.md Phase 1 step 3, step 2: rationale-label entailment gate.
# Tuned on the mining split (data/sft_train_v5_phase1_mining.jsonl) ONLY -- see V5_LOG.md
# for the tuning trace (started at 18/396 flagged with a 2-word context window and no
# negation handling, tuned down to 2/396 flagged, both manually confirmed false positives
# on inspection -- "high-threat maneuver"/"high hostile intent" idioms, not competing
# verdicts). Applied to train+val (held-out from tuning) only for the reported gate numbers.

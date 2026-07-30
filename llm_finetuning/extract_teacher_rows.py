"""Salvage teacher-prose rows from a quota-starved dataset build.

When the Groq daily token quota dies mid-build, the output is a mix of
teacher-authored rows and templated fallbacks. Template rows are exactly
identifiable (gold_assessment's fixed summary pattern), so this pulls the
teacher rows out into a fresh file you then grow with
`build_sft_dataset.py --append --teacher-only`.

Usage:
    python llm_finetuning/extract_teacher_rows.py \
        data/sft_train_final.jsonl data/sft_train_final_val.jsonl
Rewrites the FIRST file with only teacher rows and empties the val file
(the next --append run re-splits train/val properly).
"""
import json
import re
import sys

TEMPLATE = re.compile(
    r"^The swarm is in a \S+ formation( transitioning to \S+\.|, holding steady\.)$")

def main():
    paths = sys.argv[1:]
    if not paths:
        sys.exit(__doc__)
    kept, dropped = [], 0
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                summary = json.loads(row["messages"][1]["content"])["situation_summary"]
                if TEMPLATE.match(summary):
                    dropped += 1
                else:
                    kept.append(row)
    with open(paths[0], "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    for path in paths[1:]:
        open(path, "w").close()
    print(f"Kept {len(kept)} teacher rows -> {paths[0]} (dropped {dropped} templated)")

if __name__ == "__main__":
    main()

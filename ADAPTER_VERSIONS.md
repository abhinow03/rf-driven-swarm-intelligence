# Adapter version history — what changed, and why the project moved past v2

Five QLoRA adapters exist in `adapters/` (all fine-tune `Qwen/Qwen2.5-7B-Instruct`,
same LoRA shape: r=16, alpha=32, all 7 attention/MLP projections, 3 epochs,
lr=2e-4, effective batch size 8). They differ in **training data** and in one
critical **training-loss flag**, `assistant_only_loss`. This file explains what
each one is, measured from the repo (`adapter_config.json` / `training_args.bin`
in each adapter dir, `data/*.jsonl`, and `AUDIT.md`), not from memory of how the
work was planned.

## The five adapters at a glance

| adapter | training file | rows | `assistant_only_loss` | optimizer steps | can abstain? |
|---|---|---|---|---|---|
| `qwen-swarm` (v1) | `sft_train.jsonl` | 2700 | **False** (bug) | ~1014 | no |
| `qwen-swarm-v2` | `sft_train_v2.jsonl` | 810 | **False** (bug) | ~306 | no |
| `qwen-swarm-v3a` | `sft_train_final.jsonl` | 234 | **True** (fixed) | ~88 | no |
| `qwen-swarm-v3a-nomask` | `sft_train_final.jsonl` | 234 | **False** (control) | ~88 | no |
| `qwen-swarm-v3b` | `sft_train_final_abstain.jsonl` | 270 | **True** (fixed) | ~102 | **yes, partially** |

(`sft_train_v3.jsonl`, 406 rows, exists on disk but was never used to train a
named adapter — it's an intermediate dataset iteration, superseded by the further
-curated `sft_train_final.jsonl`.)

## What `assistant_only_loss` actually controls

During training, the loss can be computed over every token the model sees (the
full templated prompt *and* the JSON answer) or only over the tokens the model
is actually supposed to produce (the JSON answer). `assistant_only_loss=True` is
the correct/intended setting for instruction fine-tuning — the model should be
graded only on what it has to generate, not on how well it reproduces a prompt
it was just handed.

**`assistant_only_loss` was `False` — a genuine bug — on every adapter trained
before this fix (`qwen-swarm` and `qwen-swarm-v2`)**, discovered and confirmed in
`AUDIT.md` sec B. Fixing it wasn't trivial: Qwen2.5-7B-Instruct's own chat
template has no `{% generation %}` markers, so naively requesting an
assistant-only mask silently returns an all-zero mask with just a warning (no
error) — `train_qlora.py` had to be changed to detect this and use `trl`'s
bundled template with the correct markers instead (commit `9c0bc89`).

## Why the project moved to v3a/v3b even though v2 scores higher

This is the honest, measured answer, not a guess:

1. **The masking bug was real and needed fixing regardless of what it cost in
   accuracy.** With `assistant_only_loss=False`, part of what v1/v2 "learned"
   during training was to reproduce the templated prompt structure, not purely
   the answer-generation task — an internal-validity problem independent of any
   benchmark score. `qwen-swarm-v3a` exists specifically to fix this.

2. **v2 structurally cannot abstain.** None of `sft_train_v2.jsonl`'s 810 rows
   (or `sft_train_final.jsonl`'s 234) ever contain a training target with
   `likely_intent="unknown"` or any "I don't know" signal — the model was never
   shown that declining an answer is an option. On inputs that are genuinely
   unanswerable (a 4-hop formation chain, a context that ends mid-transition),
   v2 doesn't decline — it confidently answers anyway
   (`abstention_rate_when_unanswerable = 0%` on every such axis, `AUDIT.md` secs
   G/H). `qwen-swarm-v3b` was built specifically to add this capability:
   `sft_train_final_abstain.jsonl` = the same 234 rows + 36 new rows whose
   target is a schema-valid "this input can't be assessed" answer
   (`likely_intent="unknown"`, `confidence_in_assessment="low"`, commit `5ddcec8`).
   This is a **capability** v2 doesn't have at all, not a number v2 merely scores
   lower on — no accuracy metric on the clean battery reflects it because the
   clean battery has no unanswerable cases by design.

3. **The masking fix, isolated cleanly, is measurably real and helps more than
   a flat number suggests.** `qwen-swarm-v3a-nomask` was trained as a pure
   control — identical data and hyperparameters to v3a, only
   `assistant_only_loss` flipped back to `False` — specifically so the masking
   effect could be measured without the v2-vs-v3a confound of also changing
   dataset size. Result (`AUDIT.md` sec L): masking is worth **+13.5 points** on
   clean in-distribution inputs and **+11 to +36 points (mean +21.8)** under the
   kind of input degradation (dropped context lines, decayed confidence,
   contradictory cues) a real deployment would actually see. The benefit *grows*
   under distribution shift — exactly the condition a fielded system needs to be
   robust to.

4. **Abstention, once added, at least partially generalizes rather than being
   pure memorization.** `qwen-swarm-v3b` abstains 100% on multi-hop chains
   deeper than anything in its training data and holds ~95–100% on a stratified
   retest — but does *not* transfer to structurally different unanswerability
   (a self-contradictory context, an out-of-vocabulary formation name), so this
   is a real but narrow, still-developing capability (`AUDIT.md` secs G/I/K).

## The cost this tradeoff carries — and it was not a clean, intentional one

Here's the part that should not be sugar-coated: switching to the smaller,
properly-masked, abstention-augmented datasets did not come for free, and the
size of the cost was **not** something anyone signed up for in advance — it was
discovered during this audit, not planned as an accepted tradeoff.

**v3a and v3a-nomask cannot currently say a threat is `low` at all** — 0 out of
55 test cases across the board — and instead say `medium` on every genuinely
low-threat (benign, steady-state) scenario. v3b does slightly better (13.3%
correct) but is still mostly wrong on this class. v2 gets this right. Multiple
angles were checked before concluding why (`AUDIT.md` secs M–P):

- Not class imbalance — `low`-threat rows are proportionally represented in
  every training file (~25–27%, matching RULES' own 26.5% share).
- Not template-fallback/memorization — `sft_train_final.jsonl` (v3a's data) is
  *more* purely teacher-written (0% template rows) than v2's data (49.9%
  template rows), the opposite of what a "v2 wins because of memorization"
  story would predict.
- What's left, by elimination: **raw example count per rule pair** — v2 has
  ~16.5 training examples per `(formation_a, formation_b)` pair, v3a/v3b have
  ~4.8 — is the best-supported explanation so far, but this was inferred by
  elimination across two data points, not proven by a controlled sweep, and a
  new session is currently testing this directly (see below).

There's also an unresolved confound the team is actively separating right now:
v2 trained for far more optimizer steps than v3a (roughly 306 vs ~88), so "more
data" and "more training steps" have been entangled the whole time. An
epoch-matched control run (`qwen-swarm-v3c`, same 234 rows as v3a but trained
for enough epochs to match v2's step count) is in progress specifically to
determine whether the `low`-threat collapse is under-training (fixable by
training longer on the same data) or a genuine data-diversity ceiling (would
require more/better data, not just more steps).

## Bottom line

v2 is the better-performing adapter **today**, on raw accuracy, full stop — see
`AUDIT.md` sec Q for the plain demo recommendation (use v2 if the capstone demo
needs a non-baseline fine-tuned model). v3a/v3b exist because they fix a real
training-correctness bug and add a real new capability (abstention) that v2
structurally cannot have — not because they were expected to out-score v2 on
this specific benchmark. The `low`-threat regression they introduced along the
way is a genuine, still partially-unexplained cost of moving to a much smaller
dataset, actively under investigation, not a known-and-accepted price that was
paid on purpose.

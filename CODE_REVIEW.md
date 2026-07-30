# Codebase Review — LLM-Driven Semantic RF Analysis for UAV Swarm Communication

*Reviewer perspective: senior ML engineer + research reviewer. Brutally honest, publication-oriented.*
*Date: 2026-06-17*

> **Scope note / assumption:** This review covers only what is physically present in the workspace: `README.md`, three Jupyter notebooks (`capstone_with_llm.ipynb`, `capstone_with eval.ipynb`, `models + data generation.ipynb`), the `evaluation/` outputs, and `visuals/`. The README describes a four-component system. **Two of the four components — the RF fingerprinting model and the EKF/multilateration tracker — have no code anywhere in this repository** (no `Conv1d/Conv2d`, no `SupConLoss`, no `KalmanFilter`, no STFT, no multilateration). I assume that code, if it exists, lives in a separate repo not shared here. Everything I say about those components is based on the README alone, and I flag it as such.

---

## 1. What the project actually is

**Stated objective (README):** Move from raw RF/I-Q signals to human-readable tactical assessments of a UAV *swarm*, via a 4-stage pipeline: (1) on-drone RF fingerprinting, (2) base-station EKF multilateration + swarm graph modeling, (3) an LLM "intent" reasoning layer, (4) a 3D/AR dashboard. The genuinely interesting framing is the three-layer reasoning model — **Perception → Kinematics → Intent** — with the LLM supplying the semantic "what does this motion *mean*" layer that conventional RF-ML pipelines lack.

**What is actually implemented in this repo:**

- **Component 3 — Swarm behavior model ("STGT"):** a Spatial-Temporal Graph Transformer. Per timestep, 6 drone positions → a proximity graph → 2× `GATv2Conv` → `global_mean_pool` (spatial). 50 timesteps of graph vectors → positional encoding → 4× `TransformerEncoderLayer` → mean-pool (temporal). Three output heads: an 8-way formation classifier + a 3-value regression head (centroid velocity, approach rate, stability). Trained with AdamW + OneCycleLR + early stopping. This part is real, complete, and reasonably well-built.
- **Component 4 — LLM interpretation layer:** sliding-window inference over a long position sequence → a rule-based "tactical context" builder → a prompt assembler → a single Groq API call (`llama-3.3-70b-versatile`, temp 0.3) returning structured JSON (threat level, intent, action, explanation). Plus an evaluation suite (6 test cases, LLM-as-judge).
- **Data generation:** fully synthetic. Procedural geometric formation templates (`get_formation_offsets`) + linear centroid motion + Gaussian noise (`generate_swarm_sequence`).
- **Visualization:** static 3D scatter snapshots of the 7 formations + a converging animation GIF. No "real-time" or "AR" dashboard exists.

**Bottom line on identity:** This is **a synthetic-data swarm-formation classifier with an LLM narration layer on top** — about half of the system the README advertises. The README reads like a system that is far more complete and far more grounded in real RF than the code supports.

### High-level codebase map

```
README.md                      ← describes 4-component system (2 components have no code here)
capstone_with_llm.ipynb        ← data gen + swarm model + LLM integration   (defines generate_swarm_sequence)
capstone_with eval.ipynb       ← near-duplicate + evaluation
models + data generation.ipynb ← near-duplicate consolidated version; ★ does NOT define
                                  generate_swarm_sequence yet calls it (would NameError standalone)
evaluation/
  ml_confusion_matrix.png      ← swarm classifier on synthetic test set
  llm_eval_summary.png         ← LLM eval (self-contradictory — see §4)
  llm_run_output.json          ← one end-to-end demo run
  sliding_window_predictions_demo.json
visuals/
  formation_snapshots.png, converging_animation.gif

Absent: requirements.txt, environment.yml, LICENSE, .gitignore, src/ package,
        tests/, saved model weights, dataset artifacts, RF/EKF code.
```

The three notebooks are largely the same code at different stages of iteration. `get_formation_offsets`, `train`, `call_llm` and others are duplicated across all three. This is the single biggest structural problem (see §3).

---

## 2. Component-by-component analysis

**Data preprocessing & generation.** Formations are hardcoded `(6,3)` offset templates scaled by a `spread` factor, translated each timestep by a random constant velocity, plus Gaussian noise; "converging" shrinks offsets by a cosine/linear ramp. The split is stratified 70/15/15 and — to their credit — **normalization statistics are computed on the training set only** and reused for val/test (they explicitly comment on why leakage matters). That awareness is good. The fatal limitation is that train, val, and test are all drawn from the *same generator with the same parameter distribution*, so the test set measures in-distribution recall, not generalization (see §4).

**Feature extraction.** Graph construction (`build_graph`) uses Euclidean-distance thresholding for edges with `(Δx,Δy,Δz,dist)` edge features and `(x,y,z)` node features. Clean and appropriate for the task. Note that `GATv2Conv` as used does not consume the edge features in the standard call, so the 4-D edge attributes may be computed but unused — worth verifying.

**Models / algorithms.** GATv2 (spatial) + Transformer (temporal) + multi-task heads is a defensible, even elegant, design for formation classification, and the inline rationale (GATv2 vs GAT, Transformer vs LSTM, multi-task regularization) shows real understanding. The "head-then-full" fine-tuning to add the 8th `transitioning` class is a legitimately good technique.

**Training pipeline.** AdamW + OneCycleLR + early stopping + gradient handling is standard and correct. `num_workers=0` everywhere (fine for Colab). Reproducibility is broken (see §4) because `generate_swarm_sequence` and the dispersed/converging branches of `get_formation_offsets` call `np.random.default_rng()` / `default_rng(seed=None)` — **unseeded** — so despite a `seed=42` comment in `generate_dataset`, the dataset is not reproducible.

**Evaluation pipeline.** Two parts. (a) `evaluate_ml_model`: classification report + confusion matrix on the synthetic v2 test set. (b) `run_llm_evaluation`: 6 hand-authored scenarios × 2 runs, scored by substring family-matching against the team's own expected intent/threat, plus an LLM-as-judge second call. The ML half is mechanically fine; the LLM half is methodologically broken (see §4).

**Inference scripts.** `sliding_window_inference` (stride-10, 50-window) → `build_tactical_context` → `build_llm_prompt` → `call_llm` is a clean, sensible real-time-style flow and is the nicest-engineered part of the LLM side.

**Utilities / config.** A central `CFG` dict is good practice — but it spawns `CFG_V2`, `predict_v2`, `model_v2`, `tm_v2`, `ts_v2` as the project iterated, and later cells depend on those as **module-level globals** rather than parameters. Fragile.

**Documentation / README.** Genuinely well-written and communicative — arguably the strongest artifact in the repo. But it materially overstates the implemented system (see §4 and §6).

---

## 3. Code quality

**Organization & modularity — weak (notebook-grade).** Everything lives in three overlapping notebooks. There is no `src/` package, no importable modules, no separation of data/model/eval. The same functions are copy-pasted across notebooks, and `models + data generation.ipynb` is internally inconsistent — it *calls* `generate_swarm_sequence` but never defines it, so it only runs if another notebook seeded the kernel first. That is a latent `NameError`.

**Naming — mixed.** Domain names are clear (`get_formation_offsets`, `build_tactical_context`). But `_v2` suffixes (`CFG_V2`, `predict_v2`, `model_v2`, `tm_v2`, `ts_v2`) are versioning-by-copy, not abstraction.

**Readability — high at the line level, low at the system level.** Comments are abundant and tutorial-quality (good for a learning artifact). But comments like "FIXED — using Groq", "missing from original code", "Paste this entire file as a new cell" reveal ad-hoc patching and that the notebooks are stitched-together cell blocks rather than a designed program.

**Reusability — low.** Nothing is importable; reuse means copy-paste, which is exactly what happened.

**Duplicated code — major.** Three notebooks, heavily overlapping; plus v1/v2 function pairs within a notebook.

**Dead / unused code — present.** Multiple "find my .npy files" diagnostic cells, Colab-install cells, `visualize_all_formations_static` "missing from original code" patches, and likely-unused graph edge features.

**Potential bugs / fragility.**
- Unseeded RNG in data generation → non-reproducible datasets.
- Cross-cell global dependencies (`model_v2`, `tm_v2`, …) → notebook must be run top-to-bottom in one specific order or it breaks.
- `regression labels` computed on **normalized** positions, so `centroid_velocity` is reported in normalized units (~0.04–0.07 in `llm_run_output.json`) while the drones actually move 3–8 m/s. The README's worked example shows `centroid_velocity: 4.3` — inconsistent with what the pipeline actually emits. The LLM is reasoning over physically meaningless magnitudes.
- `transition_to` flips between `converging`/`column` across adjacent windows in the demo output — the transition detector is noisy.
- Hardcoded paths (`swarm_data/`, `/content/drive/MyDrive/...`) and `from google.colab import userdata` hard-couple the code to Colab.

**Error handling.** Essentially only `call_llm` has a try/except (and it does fence-stripping nicely). No `response.raise_for_status()`, no retry/back-off/rate-limit handling. Everything else assumes the happy path.

**Logging / debugging.** `print()` only; no `logging`, no run config capture, no experiment tracking.

**Security — acceptable.** API key is read from Colab secrets with a `"gsk_YOUR_GROQ_KEY_HERE"` placeholder fallback. No real key is committed (verified).

---

## 4. ML & research assessment — the part a reviewer will fail you on

**Does the methodology make sense?** The *architecture* does. The *evaluation and data* do not support the claims.

**Conceptual flaw 1 — the whole system is validated on synthetic data generated by hand-coded geometry.** Each formation is a fixed template + linear motion + Gaussian noise. These classes are nearly linearly separable by construction. The confusion matrix (≈0.83–1.00 per class) therefore demonstrates that *a model can memorize the generator*, not that it can recognize real swarm behavior. There is no real RF, no real flight dynamics, no domain shift. For a project whose title is "RF Analysis … of UAV Swarm Communication," there is **no RF and no communication signal in the modeled data** — only positions.

**Conceptual flaw 2 — circular regression supervision.** The regression "pseudo-labels" (velocity, approach rate, stability) are computed analytically from the exact same position tensor the model receives as input. The network is being trained to re-derive a closed-form function of its own input. This inflates apparent multi-task performance and teaches the model nothing it couldn't compute deterministically.

**Conceptual flaw 3 — the LLM evaluation contradicts itself.** Look at `evaluation/llm_eval_summary.png`:
- Objective **Intent accuracy ≈ 0/6** and **Threat accuracy ≈ 2/6** against the team's own expected answers.
- Yet the **LLM-as-judge** (the *same* `llama-3.3-70b` model, via the *same* `call_llm`) scores the system **5.00/5 on Reasoning, Consistency, and Action**, 4.75 Factual, 4.00 Threat-calibration.

A model grading its own outputs is a textbook self-preference bias, and here it directly masks a near-total failure on the objective metric. Reporting 5/5 reasoning while objective intent accuracy is ~0 is the kind of contradiction that ends a paper review. Additionally: only **6 test cases**, **2 runs** (so "100% consistency" is statistically meaningless), and the "ground-truth" intents are the team's own assumptions, matched by loose substring containment.

**Data / evaluation leakage.** No classic train-on-test leakage in the splits (normalization is train-only — good). But there is **distributional leakage by construction**: test data is i.i.d. with training data from the same generator, so the evaluation cannot detect overfitting to the simulator. This is the more damaging form here.

**Are the metrics appropriate?** Confusion matrix / per-class F1 are appropriate *for what they measure*, but the test set is dominated by the `transitioning` class (~1,760 vs ~150 each for the others), so aggregate accuracy is inflated by class imbalance. The LLM metrics (self-judge, n=2, substring match) are not appropriate.

**Is it reproducible?** No, in current form: unseeded data RNG, no `requirements.txt`/pinned versions, hardcoded Colab paths, no committed weights or dataset artifacts, and a notebook that won't run standalone.

**Is it scientifically sound / publishable?** Not yet. The concept is publishable *in principle* (LLM semantic layer over swarm kinematics is a fresh framing), but the current evidence base — synthetic geometry, circular regression labels, a self-contradictory self-judged LLM eval, and two of four components missing — would not survive peer review or a serious advisor. It is a strong *prototype/demo*, not a *result*.

**Where is the novelty, and what's needed?** The novelty is the framing (perception→kinematics→intent with an LLM intent layer) and the structured-JSON→LLM tactical-context pipeline. To make it real: validate on real or physics-grounded swarm trajectories, get an *independent* LLM (or human raters) to judge the intent layer, expand to a proper test battery (dozens–hundreds of scenarios with adversarial/ambiguous cases), and actually implement (or include) the RF and tracking front-end the title promises.

---

## 5. Software engineering quality

**Project structure — poor for collaboration/release.** No package, no requirements, no license, no tests, no CI, no `.gitignore`. A new contributor cannot `pip install -r` and run anything; they must reconstruct Colab state by hand.

**Scalability — limited.** Fixed at exactly 6 drones (`(6,3)` hardcoded throughout), 50-step windows, 8 classes. Changing drone count or formations means editing multiple hardcoded literals across notebooks.

**Extensibility — constrained** by the v1/v2 copy-versioning and global-state coupling. Adding a 9th class would again mean head surgery and another `_v3` lineage.

**Ease of collaboration — low.** Notebooks are merge-hostile (JSON + embedded outputs), and three overlapping copies guarantee divergence. (The repo even shows the team pasting cells between notebooks.)

**Ease of deployment — low.** Colab-coupled (`google.colab.userdata`, Drive paths). README mentions ONNX/edge as the target but there is no export code.

**GitHub/open-source readiness — not yet.** Compelling README, but missing license, dependencies, reproducible entry point, and a clean source layout. Embedded notebook outputs bloat the repo.

---

## 6. Weaknesses (with severity and fixes)

**Critical**
1. **LLM eval is self-judged and self-contradictory** (5/5 reasoning vs ~0 objective intent accuracy). *Why it matters:* invalidates the project's headline "it works" claim. *Fix:* drop self-judging; use an independent model and/or human raters, report the objective metric prominently, expand to ≥50 scenarios, run ≥5–10 samples per case for real consistency stats.
2. **Validation entirely on synthetic, self-generated geometry** (distributional leakage). *Why:* near-perfect accuracy is uninformative about reality. *Fix:* obtain real or physics-simulated swarm trajectories; hold out *generators/parameter regimes*, not just samples; report cross-distribution performance.
3. **README claims components that have no code here** (RF fingerprinting V4, EKF multilateration, QLoRA fine-tune, ONNX, AR dashboard). *Why:* a reviewer who opens the repo will see the gap immediately and distrust the rest. *Fix:* either include the code or clearly mark these as "planned / in a separate repo" in the README.

**Major**
4. **Three heavily duplicated notebooks; one won't run standalone** (`generate_swarm_sequence` undefined in `models + data generation.ipynb`). *Fix:* collapse into one `src/` package + thin notebooks that import it.
5. **Non-reproducible data generation** (unseeded RNG). *Fix:* thread a single seeded `Generator` through all sampling; commit a `requirements.txt` with pinned versions.
6. **Circular regression pseudo-labels** computed from the input itself. *Fix:* drop them, or derive targets from independent metadata (true simulated velocity/spread), and report regression error honestly.
7. **Units bug:** regression outputs are in normalized space but presented/interpreted as physical m/s; README example (4.3 m/s) disagrees with actual output (~0.05). *Fix:* de-normalize before reporting; assert physical ranges.

**Moderate**
8. **Class imbalance** (transitioning ≫ others) inflates aggregate accuracy. *Fix:* balance the test set or report macro-F1 / per-class only; the README already does per-class, so lead with that.
9. **Cross-cell global coupling** (`model_v2`, `tm_v2`, `ts_v2`, …). *Fix:* pass state explicitly; encapsulate in a class or config object.
10. **No tests, no logging, no experiment tracking.** *Fix:* add unit tests for graph construction / label computation, swap `print` for `logging`, log run configs.
11. **Colab-hardcoded paths and secrets.** *Fix:* use env vars / a config file / `pathlib` relative paths.

**Minor**
12. No `LICENSE`, no `.gitignore`; notebook outputs committed (repo bloat). *Fix:* add both; strip outputs (`nbstripout`) or move heavy artifacts out.
13. No API retry/back-off in `call_llm`. *Fix:* add `raise_for_status` + exponential back-off.
14. Possible unused graph edge features in `GATv2Conv`. *Fix:* either feed `edge_attr` into an edge-aware conv or drop the computation.

---

## 7. Strengths

1. **Genuinely interesting framing.** Perception → Kinematics → Intent, with an LLM supplying the semantic layer, is a fresh and well-motivated angle on RF-ML.
2. **Sound swarm architecture and clear justification.** GATv2-vs-GAT, Transformer-vs-LSTM, multi-task regularization reasoning is correct and shows real ML understanding.
3. **Correct handling of normalization leakage** — train-only stats, with an explicit comment explaining *why*. Many students get this wrong.
4. **Head-then-full fine-tuning** to add the `transitioning` class is a legitimately good, non-obvious technique.
5. **Clean inference design:** sliding-window → tactical-context → structured-JSON → LLM is a tidy, deployable-shaped pipeline.
6. **Iterative self-correction of the eval** (semantic intent families, hallucination check replacing a cruder GPT-based eval) shows research maturity — the instinct is right even if the execution still has the self-judge flaw.
7. **Honest failure documentation** in the README (CS-SEI memorization, DroneRF underfitting, the USRP "ring manifold" hardware limit). This kind of negative-result honesty is exactly what good research writing looks like.
8. **Excellent README/communication** — structure, tables, worked examples, roadmap.
9. **Good visualizations** — clear 3D formation snapshots and an animation.
10. **Central `CFG` config** and tidy, well-commented model code at the cell level.

---

## 8. Publication-readiness scores (1–10)

| Dimension | Score | One-line justification |
|---|---|---|
| Research novelty | **5** | Fresh framing (LLM intent layer over swarm kinematics); implemented novelty is thin. |
| Technical depth | **5** | Swarm model is solid; but ~half the claimed system is unimplemented and data is trivial. |
| Code quality | **4** | Notebook-grade, heavy duplication, global coupling, hardcoded paths. |
| Experimental rigor | **2** | In-distribution synthetic only; self-contradictory self-judged LLM eval; n=6 cases. |
| Reproducibility | **3** | Unseeded RNG, no deps, Colab-coupled, artifacts/weights absent. |
| Scalability | **3** | Hardcoded to 6 drones / 50 steps / 8 classes. |
| Maintainability | **3** | Three diverging copies; v1/v2 versioning-by-copy. |
| Publication potential | **2** | Strong concept, but evidence base would not survive review as-is. |

---

## 9. Deliverables

### A. Executive summary
The project pursues a genuinely compelling idea — an LLM "intent" layer that turns swarm kinematics into human-readable tactical assessments — and the swarm-formation model behind it (GATv2 + Transformer, multi-task heads, sensible training) is competently built and well-explained. But the repository delivers roughly half of the system the README advertises (no RF-fingerprinting, EKF, fine-tuning, ONNX, or dashboard code is present), and the scientific evidence is weak: everything is validated on hand-generated synthetic geometry, the regression labels are circular, and the flagship LLM evaluation is judged by the same LLM it is grading — producing a 5/5 self-score that directly contradicts a near-zero objective intent accuracy in the team's own summary figure. As an engineered prototype and a portfolio story it is promising; as a research result it is not yet defensible, and as a software project it is notebook-grade and not collaboration- or deployment-ready.

### B. Top 10 issues to fix first
1. Replace the self-judging LLM evaluation with an independent judge/human rating, and lead with the objective metric.
2. Stop validating only on self-generated synthetic data; introduce real or physics-grounded trajectories and out-of-distribution test regimes.
3. Reconcile the README with the code — mark RF/EKF/QLoRA/ONNX/dashboard as planned or include them.
4. Consolidate three notebooks into one `src/` package; eliminate the undefined-function notebook.
5. Make data generation reproducible (single seeded RNG end-to-end) and add a pinned `requirements.txt`.
6. Fix the normalized-vs-physical units bug in the regression outputs the LLM consumes.
7. Replace circular regression pseudo-labels with independent targets (or drop them).
8. Report macro-F1 / per-class metrics and address the transitioning-class imbalance.
9. Remove cross-cell global coupling; pass state explicitly.
10. Add `LICENSE`, `.gitignore`, strip committed notebook outputs, and remove dead diagnostic cells.

### C. Top 10 strengths
1. Compelling, well-motivated perception→kinematics→intent framing.
2. Sound, well-justified GATv2 + Transformer swarm architecture.
3. Correct, leakage-aware (train-only) normalization.
4. Smart head-then-full fine-tuning to add a class.
5. Clean sliding-window → tactical-context → structured-JSON → LLM inference design.
6. Iterative improvement of the evaluation methodology.
7. Honest documentation of dataset/hardware failures (real research maturity).
8. Excellent, well-structured README.
9. Clear 3D visualizations and animation.
10. Centralized config and readable, well-commented model code.

### D. Refactoring recommendations
- Create `src/swarm/` with `data.py`, `graph.py`, `model.py`, `train.py`, `infer.py`, `llm.py`, `eval.py`; reduce notebooks to thin demos that import these.
- Introduce a single `Config` dataclass (replace `CFG`/`CFG_V2`); make `n_drones`, `n_classes`, `window`, `stride` parameters, not literals.
- Thread one `numpy.random.Generator(seed)` through all sampling; add `set_seed()` for torch.
- Add `requirements.txt`/`environment.yml`, `LICENSE`, `.gitignore`, and `nbstripout`.
- Add `tests/` for graph construction, regression-label math, and prompt assembly.
- Replace `print` with `logging`; persist run configs + metrics (even a CSV or Weights & Biases).
- Add `raise_for_status` + retry/back-off to `call_llm`; make the model name and key configurable via env.
- De-normalize regression outputs before they enter the LLM prompt; assert physical ranges.

### E. Recommended next milestones
1. **Honest eval v2** (2–4 weeks): independent-judge + human spot-check, ≥50 scenarios incl. ambiguous/adversarial, ≥5 samples/case for variance; publish a results table with confidence intervals.
2. **Realistic data** (3–6 weeks): integrate a physics/flocking simulator (e.g., Boids/AirSim-style) or a public swarm-trajectory dataset; demonstrate cross-distribution generalization.
3. **Repo hardening** (1 week): package refactor, deps, license, tests, CI, stripped outputs.
4. **Close the README gap** (ongoing): either land the RF + EKF front-end in-repo or restructure the README around what exists.
5. **Then** the QLoRA/ONNX/dashboard roadmap items — only after the evidence base is solid.

### F. Overall verdict
- **Prototype quality:** Good — a working, demoable end-to-end synthetic pipeline with a polished narrative.
- **Engineering quality:** Below production — notebook-grade, duplicated, Colab-coupled, untested.
- **Research quality:** Weak in current form — synthetic-only validation, circular labels, and a self-judged eval that contradicts its own objective numbers.
- **GitHub portfolio quality:** Strong potential — the story and README are excellent; needs structure, license, deps, and honesty-alignment to shine.
- **Publication potential:** Not currently publishable. The idea has a paper in it, but only after real/realistic data, an independent rigorous evaluation, and the missing front-end components are in place.

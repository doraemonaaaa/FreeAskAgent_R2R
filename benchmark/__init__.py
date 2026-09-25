"""Subgoal-level instruction-following benchmark for R2R-CE (docs/subgoal_benchmark_design.md).

Foundation
  common          paths (data/<set>/{rule,llm}/...), R2R-CE / dense-gt loading,
                  trace loading, split writing, turn geometry, nDTW
  build_subgoals  FGR2R human chunks -> per-episode subgoal boundaries B_k
                  (data/<set>/subgoals.json) -- every variant and metric builds on this

Variants (data/<set>/rule/ -- minimal pairs, comparable to ORIG directly)
  build_flip      FLIP: the left/right of the first sub-instruction reversed; the turn is
                  at the start, so every system is scored on the same episodes
                  (data/val_unseen/, full val_unseen, v1-3 start heading)
  build_goalonly  GOAL-ONLY: only the last sub-instruction kept

Variants (data/<set>/llm/ -- LLM rewrites, only comparable to their own A1 control)
  build_paraphrase  PARAPHRASE arms para_id / para_terse / para_natural / para_lm_shift

Scoring and tooling
  metrics         SGCR(-eff), SGCR@k,
                  plus the FLIP, GOAL-ONLY and paired-agreement evaluators; bootstrap 95% CI
  summarize       recompute every run into results/summary.json (REPORT.md source)
  selfcheck       noisy reference-path agent, r_b calibration
  compact_trace   shrink a runner output dir to the fields the metrics read
  canav, awarevln adapters: build each system's inputs, import its trajectories
"""
